"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

import os
import sys
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from cleverhans.loss import LossCrossEntropy
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf_pr_mnist import train, model_eval, tf_model_load
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
flags.DEFINE_integer('nb_epochs', 10000, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
flags.DEFINE_bool('backprop_through_attack', False,
                  ('If True, backprop through adversarial example '
                   'construction process during adversarial training'))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

n_input = 28*28
numColorOutput = 3  # Always 3 for MNIST and Fashion MNIST
type = 'single'
#type = 'multiple'
mode = "nonpr"
# mode = "train_test"
file = "/home/labiai/Jiacang/Experiments/tmp/data/mnist"
save_dir = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/mnist_'+str(FLAGS.nb_epochs)+'_'+mode+'_'+str(numColorOutput)+'output/'
# file = "/home/labiai/Jiacang/Experiments/tmp/data/fashion_mnist"
# save_dir = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/fmnist_'+str(FLAGS.nb_epochs)+'_'+mode+'_'+str(numColorOutput)+'output/'

print("file:", file)
filename = 'network'

save_dir2 = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/prmodel_mnist_'+str(numColorOutput)+'output/'
filename2 = 'network'
model_path2 = os.path.join(save_dir2, filename2)
n_input = 784  # MNIST data input (img shape: 28*28)

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None,
                   label_smoothing=0.1):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    #sess = tf.Session(config=tf.ConfigProto(**config_args))
    sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 1}))

    # Get MNIST test data
    x_train, y_train, x_test, y_test = data_mnist(file, train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)
    # Use Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]

    rng = np.random.RandomState([2017, 8, 30])

    ################ color training initialization ####################

    color_training_epochs = 5000
    color_learning_rate = 0.1
    colorCategory = [
        [0.0, 0.4],  # Black
        [0.3, 0.7],  # Grey
        [0.6, 1.0]  # White
    ]

    numColorInput = 1
    #numColorOutput = len(colorCategory)

    color_x = tf.placeholder(tf.float32, [None, numColorInput])  # mnist data image of shape 28*28=784
    color_y = tf.placeholder(tf.float32, [None, numColorOutput])  # 0-9 digits recognition => 10 classes

    # Set model weights
    color_W = tf.Variable(tf.zeros([numColorInput, numColorOutput]))
    color_b = tf.Variable(tf.zeros([numColorOutput]))
    color_pred_out = tf.matmul(color_x, color_W) + color_b  # Softmax

    color_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=color_pred_out, labels=color_y))
    # Gradient Descent
    color_optimizer = tf.train.GradientDescentOptimizer(color_learning_rate).minimize(color_cost)

    # Test model
    color_argmax = tf.argmax(color_pred_out, 1)
    color_correct_prediction = tf.equal(tf.argmax(color_pred_out, 1), tf.argmax(color_y, 1))
    # Calculate accuracy
    color_accuracy = tf.reduce_mean(tf.cast(color_correct_prediction, tf.float32))

    # Graph for re-generating the original image into a new image by using trained color model
    pr_model_x = tf.placeholder(tf.float32, [None, n_input, numColorInput])  # mnist data image of shape 28*28=784
    pr_model_W = tf.placeholder(tf.float32,
                                      [None, numColorInput, numColorOutput])  # mnist data image of shape 28*28=784
    pr_model_b = tf.placeholder(tf.float32,
                                      [None, numColorInput, numColorOutput])  # mnist data image of shape 28*28=784
    pr_model_output = tf.one_hot(tf.argmax((tf.matmul(pr_model_x, pr_model_W) + pr_model_b), 2), numColorOutput)

    # Merge the random generated output for new image based on the colorCategory
    randomColorCategory = []
    for i in range(len(colorCategory)):
        tmp = []
        tmpRandomColorCategory = my_tf_round(
            tf.random_uniform([batch_size, n_input, 1], colorCategory[i][0], colorCategory[i][1], dtype=tf.float32), 2)
        tmp.append(tmpRandomColorCategory)
        randomColorCategory.append(tf.concat(tmp, 1))
    random_merge = tf.reshape(tf.concat(randomColorCategory, -1), [batch_size, n_input, numColorOutput])
    random_color_set = tf.reduce_sum(tf.multiply(pr_model_output, random_merge), 2)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    x = tf.reshape(random_color_set, shape=(-1, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    print(random_color_set)



    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_dir': save_dir,
        'filename': filename,
        'numColorOutput': numColorOutput
    }
    eval_params = {'batch_size': batch_size, 'numColorOutput': numColorOutput}
    fgsm_params = {
        # 'eps': 8/256,
        'eps': 1.0,
        'clip_min': 0.,
        'clip_max': 1.
    }

    #sess = tf.Session()

    def do_eval(preds, x_set, y_set, report_key, is_adv=None,
                pred2=None, c_w=None, c_b=None, pr_model_x=None, random_color_set=None,
                pr_model_W=None, pr_model_b=None, pr_model_output=None, ae=None
                ):
        acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params,
                         pred2=pred2, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                         pr_model_W=pr_model_W, pr_model_b=pr_model_b, pr_model_output=pr_model_output, is_adv=is_adv,
                         ae=ae
                         )
        setattr(report, report_key, acc)
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples: %0.4f' % (report_text, acc))

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            tf.global_variables_initializer().run()
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "CleverHans may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        ################# color training ####################
        print("Trying to load pr model from: "+model_path2)
        if os.path.exists(model_path2 + ".meta"):
            tf_model_load(sess, model_path2)
            c_w, c_b = sess.run([color_W, color_b])
            print("Load color trained model in training")
        else:
            # Training the color
            for epoch in range(color_training_epochs):
                outputColorY = []
                p1 = np.random.random(100)
                for i in range(len(p1)):
                    outputOverlapColorY = []
                    for j in range(len(colorCategory)):
                        if p1[i] >= colorCategory[j][0] and p1[i] <= colorCategory[j][1]:
                            colorIndexSeq = []
                            for k in range(len(colorCategory)):
                                if j == k:
                                    colorIndexSeq.append(1)
                                else:
                                    colorIndexSeq.append(0)
                            outputOverlapColorY.append(colorIndexSeq)
                            #break

                    # Randomly choose the output for color Y if the outputOverlapColorY has more than 1 item
                    outputColorY.append(outputOverlapColorY[np.random.randint(0, len(outputOverlapColorY))])

                inputColorX = p1.reshape(100, 1)
                _, c, c_w, c_b = sess.run([color_optimizer, color_cost, color_W, color_b],
                                          feed_dict={color_x: inputColorX, color_y: outputColorY})
                avg_cost = c

                # Evaluating color model
                outputColorY = []
                p1 = np.random.random(100)
                # Generate output for random color inputs (test case)
                for i in range(len(p1)):
                    for j in range(len(colorCategory)):
                        outputOverlapColorY = []
                        if p1[i] >= colorCategory[j][0] and p1[i] <= colorCategory[j][1]:
                            colorIndexSeq = []
                            for k in range(len(colorCategory)):
                                if j == k:
                                    colorIndexSeq.append(1)
                                else:
                                    colorIndexSeq.append(0)
                            outputOverlapColorY.append(colorIndexSeq)
                            break

                    # Randomly choose the output for color Y if the outputOverlapColorY has more than 1 item
                    outputColorY.append(outputOverlapColorY[np.random.randint(0, len(outputOverlapColorY))])

                inputColorX = p1.reshape(100, 1)
                # print(random_xs)
                acc, argmax = sess.run([color_accuracy, color_argmax], feed_dict={color_x: inputColorX,
                                                                                  color_y: outputColorY})
                print("Epoch:", '%04d' % (epoch + 1) + "/" + str(color_training_epochs) + ", Cost= " + \
                      "{:.9f}".format(avg_cost) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc) + " ")
                # print(c_w)

            with tf.device('/CPU:0'):
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
                # Since training PR model is fast, we do not have to save multiple sessions for this
                save_path = os.path.join(save_dir2, filename2)
                saver.save(sess, save_path)
        ##################### end of color training ------------------------------



    ################# model training ####################
    if clean_train:
        model = ModelBasicCNN('model1', nb_classes, nb_filters)
        preds = model.get_logits(x)
        loss = LossCrossEntropy(model, smoothing=label_smoothing)

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        saveFileNum = 50
        saveFileNum = 500
        # saveFileNum = 1000
        model_path = os.path.join(save_dir, filename+"-"+str(saveFileNum))
        fgsm = FastGradientMethod(model, sess=sess)
        # fgsm = BasicIterativeMethod(model, sess=sess)
        # fgsm = MomentumIterativeMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_logits(adv_x)

        def evaluate():
            do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False,
                    pred2=preds, c_w=c_w, c_b=c_b,
                    pr_model_x=pr_model_x, random_color_set=random_color_set,
                    pr_model_W=pr_model_W, pr_model_b=pr_model_b)
            #do_eval(preds, x_test, y_test, 'clean_train_adv_eval', True,
                    #pred2=preds, c_w=c_w, c_b=c_b, ae=adv_x,
                    #pr_model_x=pr_model_x, random_color_set=random_color_set,
                    #pr_model_W=pr_model_W, pr_model_b=pr_model_b, pr_model_output=pr_model_output
                    #)

        print("Trying to load trained model from: "+model_path)
        if os.path.exists(model_path + ".meta"):
            tf_model_load(sess, model_path)
            print("Load trained model")
        else:
            train(sess, loss, x, y, x_train, y_train, evaluate=evaluate,
                  args=train_params, rng=rng, var_list=model.get_params(), save=True,
                  c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                  pr_model_W=pr_model_W, pr_model_b=pr_model_b
                  )

        # Calculate training error
        if testing:
            do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

        # Evaluate the accuracy of the MNIST model on adversarial examples
        do_eval(preds, x_test, y_test, 'clean_train_adv_eval', True,
                pred2=preds, c_w=c_w, c_b=c_b, ae=adv_x,
                pr_model_x=pr_model_x, random_color_set=random_color_set,
                pr_model_W=pr_model_W, pr_model_b=pr_model_b
                )

        # Calculate training error
        if testing:
            do_eval(preds_adv, x_train, y_train, 'train_clean_train_adv_eval')

def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    with tf.device('/device:GPU:0'):
        tf.app.run()
