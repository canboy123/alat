"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
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
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.loss import LossCrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf_multiple_pr_mnist import train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
FLAGS = flags.FLAGS

flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
flags.DEFINE_integer('nb_epochs', 10000, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_integer('source_samples', 10, 'Nb of test inputs to attack')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_string('model_path', os.path.join("models", "mnist"),
                    'Path to save or load the model file')
flags.DEFINE_integer('attack_iterations', 1000,
                     'Number of iterations to run attack; 1000 is good')
flags.DEFINE_boolean('targeted', False,
                     'Run the tutorial in targeted mode?')

n_input = 28*28
numColorOutput = 3  # Always 3 for MNIST and Fashion MNIST
type = 'single'
type = 'multiple'
mode = "normal"
mode = "train_test"
file = "/home/labiai/Jiacang/Experiments/tmp/data/mnist"
save_dir = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/mnist_'+str(FLAGS.nb_epochs)+'_'+mode+'_'+str(numColorOutput)+'output/'
file = "/home/labiai/Jiacang/Experiments/tmp/data/fashion_mnist"
save_dir = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/fmnist_'+str(FLAGS.nb_epochs)+'_'+mode+'_'+str(numColorOutput)+'output/'


print("file:", file)
filename = 'network'

def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def mnist_tutorial_cw(train_start=0, train_end=60000, test_start=0,
                      test_end=10000, viz_enabled=True, nb_epochs=6,
                      batch_size=128, source_samples=10,
                      learning_rate=0.001, attack_iterations=100,
                      model_path=os.path.join("models", "mnist"),
                      targeted=True):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param source_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    set_log_level(logging.DEBUG)

    # Get MNIST test data
    x_train, y_train, x_test, y_test = data_mnist(file, train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Obtain Image Parameters
    img_rows, img_cols, nchannels = x_train.shape[1:4]
    nb_classes = y_train.shape[1]
    nb_filters = 64

    ################ color training initialization ####################

    color_training_epochs = 5000
    color_learning_rate = 0.1
    colorCategory = [
        [0.0, 0.4],  # Black
        [0.3, 0.7],  # Grey
        [0.6, 1.0]  # White
    ]

    save_dir2 = '/home/labiai/Jiacang/Experiments/tmp/pr/'+type+'/prmodel_'+str(FLAGS.nb_epochs)+'_'+str(len(colorCategory))+'_output_model/'
    filename2 = 'network'
    model_path2 = os.path.join(save_dir2, filename2)

    numOfPRModel = 20
    minColorEpoch = 300
    maxColorEpoch = 3000

    numColorInput = 1

    color_x = tf.placeholder(tf.float32, [None, numColorInput])  # mnist data image of shape 28*28=784
    color_y = tf.placeholder(tf.float32, [None, numColorOutput])  # 0-9 digits recognition => 10 classes

    # Set multiple models' weights and biases
    color_W = {}
    color_b = {}
    color_pred_out = {}
    color_cost = {}
    color_optimizer = {}
    color_argmax = {}
    color_correct_prediction = {}
    color_accuracy = {}
    for i in range(numOfPRModel):
        color_W["w" + str(i)] = tf.Variable(tf.random_normal([numColorInput, numColorOutput]))
        color_b["b" + str(i)] = tf.Variable(tf.random_normal([numColorOutput]))
        color_pred_out["out" + str(i)] = tf.matmul(color_x, color_W["w" + str(i)]) + color_b["b" + str(i)]  # Softmax
        color_cost["cost" + str(i)] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=color_pred_out["out" + str(i)], labels=color_y))
        # Gradient Descent
        color_optimizer["opt" + str(i)] = tf.train.GradientDescentOptimizer(color_learning_rate).minimize(
            color_cost["cost" + str(i)])

        # Test model
        color_argmax["argmax" + str(i)] = tf.argmax(color_pred_out["out" + str(i)], 1)
        color_correct_prediction["pred" + str(i)] = tf.equal(tf.argmax(color_pred_out["out" + str(i)], 1),
                                                             tf.argmax(color_y, 1))
        # Calculate accuracy
        color_accuracy["acc" + str(i)] = tf.reduce_mean(tf.cast(color_correct_prediction["pred" + str(i)], tf.float32))


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
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    x = tf.reshape(random_color_set, shape=(-1, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

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
            # Training the PR model
            c_w = {}
            c_b = {}
            for modelcount in range(numOfPRModel):
                color_training_epochs = np.random.randint(minColorEpoch, maxColorEpoch)
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
                                # break

                        # Randomly choose the output for color Y if the outputOverlapColorY has more than 1 item
                        outputColorY.append(outputOverlapColorY[np.random.randint(0, len(outputOverlapColorY))])

                    inputColorX = p1.reshape(100, 1)
                    _, c, _c_w, _c_b = sess.run(
                        [color_optimizer["opt" + str(modelcount)], color_cost["cost" + str(modelcount)],
                         color_W["w" + str(modelcount)], color_b["b" + str(modelcount)]],
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
                    acc, argmax = sess.run(
                        [color_accuracy["acc" + str(modelcount)], color_argmax["argmax" + str(modelcount)]],
                        feed_dict={color_x: inputColorX, color_y: outputColorY})
                print(str(modelcount + 1) + ") Epoch:",
                          '%04d' % (epoch + 1) + "/" + str(color_training_epochs) + ", Cost= " + \
                          "{:.9f}".format(avg_cost) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc) + " ")

                c_w["w" + str(modelcount)] = _c_w
                c_b["b" + str(modelcount)] = _c_b

                # print(c_w)

            save_path = os.path.join(save_dir2, filename2)
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            ##################### end of color training ------------------------------


    # Define TF model graph
    model = ModelBasicCNN('model1', nb_classes, nb_filters)
    preds = model.get_logits(x)
    loss = LossCrossEntropy(model, smoothing=0.1)
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        #'train_dir': os.path.join(*os.path.split(model_path)[:-1]),
        #'filename': os.path.split(model_path)[-1],
        'train_dir': save_dir,
        'filename': filename,
        'numColorOutput': numColorOutput
    }

    ################# model training ####################
    rng = np.random.RandomState([2017, 8, 30])
    saveFileNum = 50
    # saveFileNum = 500
    # saveFileNum = 1000
    model_path = os.path.join(save_dir, filename+"-"+str(saveFileNum))
    # check if we've trained before, and if we have, use that pre-trained model
    print("Trying to load trained model from: "+model_path)
    if os.path.exists(model_path + ".meta"):
        tf_model_load(sess, model_path)
    else:
        train(sess, loss, x, y, x_train, y_train,
              args=train_params, rng=rng, save=True,
              c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
              pr_model_W=pr_model_W, pr_model_b=pr_model_b
              )

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size, 'numColorOutput': numColorOutput}
    '''
    accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params,
                       pred2=preds, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                       pr_model_W=pr_model_W, pr_model_b=pr_model_b)
    assert x_test.shape[0] == test_end - test_start, x_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy
    '''

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
    print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
          ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    cw = CarliniWagnerL2(model, back='tf', sess=sess)

    if viz_enabled:
        assert source_samples == nb_classes
        idxs = [np.where(np.argmax(y_test, axis=1) == i)[0][0]
                for i in range(nb_classes)]
    if targeted:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, nb_classes, img_rows, img_cols,
                          nchannels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = np.array(
                [[instance] * nb_classes for instance in x_test[idxs]],
                dtype=np.float32)
        else:
            adv_inputs = np.array(
                [[instance] * nb_classes for
                 instance in x_test[:source_samples]], dtype=np.float32)

        one_hot = np.zeros((nb_classes, nb_classes))
        one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

        adv_inputs = adv_inputs.reshape(
            (source_samples * nb_classes, img_rows, img_cols, nchannels))
        adv_ys = np.array([one_hot] * source_samples,
                          dtype=np.float32).reshape((source_samples *
                                                     nb_classes, nb_classes))
        yname = "y_target"
    else:
        if viz_enabled:
            # Initialize our array for grid visualization
            grid_shape = (nb_classes, 2, img_rows, img_cols, nchannels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            adv_inputs = x_test[idxs]
            adv_inputs = x_test
        else:
            adv_inputs = x_test[:source_samples]
            adv_inputs = x_test

        adv_ys = None
        yname = "y"

    cw_params = {'binary_search_steps': 1,
                 'max_iterations': attack_iterations,
                 'learning_rate': 0.1,
                 'batch_size': source_samples * nb_classes if
                 targeted else source_samples,
                 'initial_const': 10}

    adv2 = cw.generate(x, **cw_params)
    cw_params[yname] = adv_ys
    adv = None
    # adv = cw.generate_np(adv_inputs, **cw_params)

    eval_params = {'batch_size': np.minimum(nb_classes, source_samples), 'numColorOutput': numColorOutput}
    if targeted:
        adv_accuracy = model_eval(
            sess, x, y, preds, adv, adv_ys, args=eval_params)
    else:
        if viz_enabled:
            adv_accuracy = model_eval(sess, x, y, preds, adv, y_test[
                           idxs], args=eval_params)
        else:
            #adv_accuracy = model_eval(sess, x, y, preds, adv, y_test[
            #               :source_samples], args=eval_params)
            adv_accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params,
                                       pred2=preds, c_w=c_w, c_b=c_b, pr_model_x=pr_model_x, random_color_set=random_color_set,
                                       pr_model_W=pr_model_W, pr_model_b=pr_model_b, is_adv=True,
                                       ae=adv2
                                   )

    if viz_enabled:
        for j in range(nb_classes):
            if targeted:
                for i in range(nb_classes):
                    grid_viz_data[i, j] = adv[i * nb_classes + j]
            else:
                grid_viz_data[j, 0] = adv_inputs[j]
                grid_viz_data[j, 1] = adv[j]

        print(grid_viz_data.shape)

    print('--------------------------------------')
    print("load save file: ", saveFileNum)
    # Compute the number of adversarial examples that were successfully found
    print('Test with adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_tutorial_cw(viz_enabled=FLAGS.viz_enabled,
                      nb_epochs=FLAGS.nb_epochs,
                      batch_size=FLAGS.batch_size,
                      source_samples=FLAGS.source_samples,
                      learning_rate=FLAGS.learning_rate,
                      attack_iterations=FLAGS.attack_iterations,
                      model_path=FLAGS.model_path,
                      targeted=FLAGS.targeted)


if __name__ == '__main__':

    tf.app.run()
