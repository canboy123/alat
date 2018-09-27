from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
from keras import backend
from keras.datasets import cifar10
from keras.utils import np_utils

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
import logging
import numpy as np
import os
import sys
lib_path = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.append(lib_path)

from cleverhans.attacks import FastGradientMethod,BasicIterativeMethod,MomentumIterativeMethod
from cleverhans.utils_keras import cnn_model, vgg_model
from cleverhans.utils_tf_pr_cifar10 import model_train, model_eval, batch_eval, tf_model_load
from cleverhans.utils import set_log_level

FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', '/home/labiai/Jiacang/Experiments/tmp/pr/cifar', 'Directory storing the saved model.')
flags.DEFINE_string(
    'filename', 'cifar10.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_epochs', 10000, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')

n_input = 3072  # MNIST data input (img shape: 28*28)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

numColorOutput = 3
type = 'single'
#type = 'multiple'
mode = "nonpr"
# mode = "nonpr_vgg"
# mode = "train_test"
save_dir = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/cifar10_'+str(FLAGS.nb_epochs)+'_'+mode+'_'+str(numColorOutput)+'output/'
filename = 'network'

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def data_cifar10():
    """
    Preprocess CIFAR10 dataset
    :return:
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def main(argv=None):
    """
    CIFAR10 CleverHans tutorial
    :return:
    """

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get CIFAR10 test data
    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert Y_train.shape[1] == 10.
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)


    ################ color training initialization ####################

    color_training_epochs = 10000
    color_learning_rate = 0.1
    accepted_range = np.round(1/numColorOutput, 4)
    colorCategory = [
        [0.0, 0.4],  # Black
        [0.3, 0.7],  # Grey
        [0.6, 1.0]  # White
    ]
    #for i in range(numColorOutput):
    #    colorCategory.append([np.round((i*accepted_range), 4), np.round(((i+1)*accepted_range), 4)])

    print(colorCategory)

    save_dir2 = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/prmodel_'+str(FLAGS.nb_epochs)+'_'+str(numColorOutput)+'_output_model/'
    filename2 = 'network'
    model_path2 = os.path.join(save_dir2, filename2)
    numColorInput = 1

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
            tf.random_uniform(tf.shape(pr_model_x), colorCategory[i][0], colorCategory[i][1], dtype=tf.float32), 2)
        tmp.append(tmpRandomColorCategory)
        randomColorCategory.append(tf.concat(tmp, 1))
    random_merge = tf.reshape(tf.concat(randomColorCategory, -1), [-1, n_input, numColorOutput])
    random_color_set = tf.reduce_sum(tf.multiply(pr_model_output, random_merge), 2)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    x = tf.reshape(random_color_set, shape=(-1, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))

    with sess.as_default():
        if hasattr(tf, "global_variables_initializer"):
            tf.global_variables_initializer().run()
        else:
            warnings.warn("Update your copy of tensorflow; future versions of "
                          "CleverHans may drop support for this version.")
            sess.run(tf.initialize_all_variables())

        ################# color training ####################
        print("Trying to load pr model from: " + model_path2)
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

    # Define TF model graph
    model = cnn_model(img_rows=32, img_cols=32, channels=3)
    # model = vgg_model(img_rows=32, img_cols=32, channels=3)
    predictions = model(x)
    print("Defined TensorFlow model graph.")
    saveFileNum = 50
    # saveFileNum = 500
    # saveFileNum = 1000
    model_path = os.path.join(save_dir, filename+"-"+str(saveFileNum))

    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    fgsm = FastGradientMethod(model)
    fgsm = BasicIterativeMethod(model)
    fgsm = MomentumIterativeMethod(model)
    adv_x = fgsm.generate(x, eps=8/256)
    eval_params = {'batch_size': FLAGS.batch_size, 'n_input': n_input, 'numColorOutput': numColorOutput}

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': FLAGS.batch_size, 'numColorOutput': numColorOutput}
        accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                              args=eval_params, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                                pr_model_W=pr_model_W, pr_model_b=pr_model_b)
        #accuracy_adv = model_eval(sess, x, y, predictions, X_test,
        #                          Y_test, args=eval_params, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
        #                        pr_model_W=pr_model_W, pr_model_b=pr_model_b, is_adv=True, ae=adv_x)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))
        #print('Test accuracy on adversarial examples: ' + str(accuracy_adv))

    # Train an CIFAR10 model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'train_dir': save_dir,
        'filename': filename,
        'numColorOutput': numColorOutput
    }

    print("Trying to load trained model from: " + model_path)
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
        print("Load trained model")
    else:
        model_train(sess, x, y, predictions, X_train, Y_train,
              evaluate=evaluate, args=train_params, save=True,
              c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
              pr_model_W=pr_model_W, pr_model_b=pr_model_b)

    X_test_adv, = batch_eval(sess, [x], [adv_x], [X_test], args=eval_params)
    # assert X_test_adv.shape[0] == 10000, X_test_adv.shape


    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, x, y, predictions, X_test, Y_test,
                          args=eval_params, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                                pr_model_W=pr_model_W, pr_model_b=pr_model_b, is_adv=True, ae=adv_x)
    print('Test accuracy on adversarial examples: ' + str(accuracy))

if __name__ == '__main__':
    app.run()
