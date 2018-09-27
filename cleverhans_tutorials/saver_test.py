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
from cleverhans.utils_tf import train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import ModelBasicCNN

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
flags.DEFINE_integer('nb_epochs', 1, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
flags.DEFINE_bool('backprop_through_attack', False,
                  ('If True, backprop through adversarial example '
                   'construction process during adversarial training'))

file = "/tmp/data/fasion-mnist"
save_dir = '/tmp/pr/fmnist_'+str(FLAGS.nb_epochs)+'/'
filename = 'network'
n_input = 784  # MNIST data input (img shape: 28*28)

with tf.device('/device:GPU:0'):
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
        sess = tf.Session(config=tf.ConfigProto(**config_args))

        # Get MNIST test data
        x_train, y_train, x_test, y_test = data_mnist(train_start=train_start,
                                                      train_end=train_end,
                                                      test_start=test_start,
                                                      test_end=test_end)
        # Use Image Parameters
        img_rows, img_cols, nchannels = x_train.shape[1:4]
        nb_classes = y_train.shape[1]

        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                              nchannels))
        y = tf.placeholder(tf.float32, shape=(None, nb_classes))
        color_W = tf.Variable(tf.zeros([16, 16]))
        color_b = tf.Variable(tf.zeros([16]))
        colorCategory = [
            [0.0, 0.4],  # Black
            [0.3, 0.7],  # Grey
            [0.6, 1.0]  # White
        ]

        numColorInput = 1
        numColorOutput = len(colorCategory)

        # Merge the random generated output for new image based on the colorCategory
        randomColorCategory = []
        for i in range(len(colorCategory)):
            tmp = []
            tmpRandomColorCategory = my_tf_round(
                tf.random_uniform([n_input, 1], colorCategory[i][0], colorCategory[i][1], dtype=tf.float32), 2)
            tmp.append(tmpRandomColorCategory)
            randomColorCategory.append(tf.concat(tmp, 1))
        random_merge = tf.reshape(tf.concat(randomColorCategory, -1), [n_input, numColorOutput])
        ranges = tf.Variable(np.arange(0, n_input))
        ranges = tf.reshape(ranges, [1, n_input])
        ranges = tf.cast(ranges, tf.int32)

        single_x = tf.placeholder(tf.int32, [None, n_input])
        indices = tf.reshape(tf.concat([ranges, single_x], -1), [n_input, 2])
        random_color_set = tf.gather_nd(random_merge, indices)

        with sess.as_default():
                if hasattr(tf, "global_variables_initializer"):
                    tf.global_variables_initializer().run()
                else:
                    sess.run(tf.initialize_all_variables())
                save_path = os.path.join("/tmp/test_ae/", "test")

                print(ranges.eval())
                print(indices.eval(feed_dict = {single_x: [np.arange(0, n_input), np.arange(0, n_input)]}))



        return report


    def main(argv=None):
        mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                       learning_rate=FLAGS.learning_rate,
                       clean_train=FLAGS.clean_train,
                       backprop_through_attack=FLAGS.backprop_through_attack,
                       nb_filters=FLAGS.nb_filters)


    if __name__ == '__main__':
        tf.app.run()
