# coding=utf-8
# Copyright 2020 The Edward2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a DNN on a UCI or image dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import app
from absl import flags

from edward2.experimental.auxiliary_sampling import datasets
from edward2.experimental.auxiliary_sampling.compute_metrics import ensemble_metrics
from edward2.experimental.auxiliary_sampling.deterministic_baseline.det_conv_net import det_conv_net
from edward2.experimental.auxiliary_sampling.deterministic_baseline.det_simple_net import det_simple_net
from edward2.experimental.auxiliary_sampling.res_net import res_net
from keras import backend as K
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'Name of the UCI or image dataset.')
flags.DEFINE_integer('n_ensemble', 10, 'Number of ensemble components.')
flags.DEFINE_boolean('bootstrap', True,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 3000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float(
    'epsilon', 0.01, 'Epsilon for adversarial training.' +
    'Epsilon is given as a ratio of the input range ' +
    '(e.g the adjustment is 2.55 if input range is [0,255]).' +
    'Using epsilon=0.0 or epsilon=None disables adversarial training.')
flags.DEFINE_integer('measurement_frequency', 200, 'Measurement frequency.')
flags.DEFINE_string('working_dir', '/tmp', 'Working directory.')
flags.DEFINE_integer('seed', None, 'Random seed.')
flags.DEFINE_boolean(
    'resnet', False, 'Use a ResNet for image classification.' +
    'The default is to use the LeNet5 arhitecture.' +
    'Currently only supported on cifar10.')
flags.DEFINE_boolean('batchnorm', False,
                     'Use batchnorm. Only applies when resnet is True.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)
    tf.random.set_random_seed(FLAGS.seed)

  session = tf.Session()
  K.set_session(session)
  with session.as_default():
    image_classification = FLAGS.dataset in [
        'mnist', 'fashion_mnist', 'cifar10'
    ]

    # Fetch data
    x_train, y_train, x_test, y_test = datasets.load(FLAGS.dataset, session)
    n_train = x_train.shape[0]

    # Build net
    # The output of the network is scaled by y_std to improve stability.
    if image_classification:
      num_classes = int(np.amax(y_train)) + 1

      if not FLAGS.resnet:
        neural_net = det_conv_net(x_train.shape[1:], num_classes)
      else:
        neural_net = res_net(
            n_train,
            x_train.shape[1:],
            num_classes,
            batchnorm=FLAGS.batchnorm,
            variational=False)

        def schedule_fn(epoch):
          """Learning rate schedule function."""
          rate = FLAGS.learning_rate
          if epoch > 180:
            rate *= 0.5e-3
          elif epoch > 160:
            rate *= 1e-3
          elif epoch > 120:
            rate *= 1e-2
          elif epoch > 80:
            rate *= 1e-1
          return float(rate)

        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule_fn)
    else:
      neural_net = det_simple_net(x_train.shape[1:],
                                  np.std(y_train, axis=0) + 1e-10)

    # Losses and metrics
    def negative_log_likelihood(y, rv_y):
      del rv_y  # unused arg
      return -neural_net.output.log_prob(tf.squeeze(
          y)) if image_classification else -neural_net.output.log_prob(y)

    def accuracy(y_true, y_sample):
      del y_sample  # unused arg
      return tf.equal(
          tf.argmax(input=neural_net.output.logits, axis=1),
          tf.cast(tf.squeeze(y_true), tf.int64))

    def mse(y_true, y_sample):
      """Mean-squared error."""
      del y_sample  # unused arg
      return tf.square(neural_net.output.loc - y_true)

    def mll(y_true, y_sample):
      """Expected conditional log-likelihood."""
      del y_sample  # unused arg
      return neural_net.output.log_prob(
          tf.squeeze(y_true)
      ) if image_classification else neural_net.output.log_prob(y_true)

    if image_classification:
      metrics = [mll, accuracy]
    else:
      metrics = [mll, mse]

    # epsilon=0.0 or epsilon=None disables adversarial training.
    if FLAGS.epsilon:
      y_true = tf.keras.Input(shape=y_train.shape[1:], name='labels')
      loss = tf.reduce_mean(-neural_net.output.log_prob(y_true))
      nn_input_tensor = neural_net.input
      grad = tf.gradients(loss, nn_input_tensor)[0]
      # It is assumed that the training data is normalized.
      adv_inputs_tensor = nn_input_tensor + FLAGS.epsilon * tf.math.sign(
          tf.stop_gradient(grad))
      adv_inputs = tf.keras.Input(tensor=adv_inputs_tensor, name='adv_inputs')
      adv_out_dist = neural_net(adv_inputs)
      adv_loss = tf.reduce_mean(-adv_out_dist.log_prob(y_true))
      optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(0.5 * loss + 0.5 * adv_loss)

    model_dir = os.path.join(FLAGS.working_dir, 'models')
    tf.io.gfile.MakeDirs(model_dir)
    ensemble_filenames = []

    ensemble_filenames = []
    for i in range(FLAGS.n_ensemble):
      # Training
      tensorboard = tf.keras.callbacks.TensorBoard(
          log_dir=FLAGS.working_dir,
          update_freq=FLAGS.batch_size * FLAGS.measurement_frequency)
      neural_net.compile(
          optimizer=tf.keras.optimizers.Adam(lr=float(FLAGS.learning_rate)),
          loss=negative_log_likelihood,
          metrics=metrics)
      session.run(tf.initialize_all_variables())

      if FLAGS.epsilon:
        for epoch in range(
            math.ceil(FLAGS.batch_size * FLAGS.training_steps / n_train)):
          print('Epoch {}'.format(epoch))
          for j in range(math.ceil(n_train / FLAGS.batch_size)):
            perm = np.random.permutation(n_train)
            session.run(
                train_op,
                feed_dict={
                    nn_input_tensor: x_train[perm[j:j + FLAGS.batch_size]],
                    y_true: y_train[perm[j:j + FLAGS.batch_size]]
                })
      else:
        # Sample bootstrap dataset
        if FLAGS.bootstrap:
          inds = np.random.choice(n_train, n_train, replace=True)
          x_sampled = x_train[inds]
          y_sampled = y_train[inds]

        neural_net.fit(
            x=x_train if not FLAGS.bootstrap else x_sampled,
            y=y_train if not FLAGS.bootstrap else y_sampled,
            batch_size=FLAGS.batch_size,
            epochs=math.ceil(FLAGS.batch_size * FLAGS.training_steps / n_train),
            validation_data=(x_test, y_test),
            validation_freq=math.ceil(FLAGS.measurement_frequency *
                                      FLAGS.batch_size / n_train),
            verbose=1,
            callbacks=[tensorboard]
            if not FLAGS.resnet else [tensorboard, lr_callback])

      ensemble_filename = os.path.join(
          model_dir, 'ensemble_component_' + str(i) + '.weights')
      ensemble_filenames.append(ensemble_filename)
      neural_net.save_weights(ensemble_filename)

    labels = tf.keras.layers.Input(shape=y_train.shape[1:])
    if image_classification:
      ll = K.function([neural_net.input, labels], [
          neural_net.output.log_prob(tf.squeeze(labels)),
          neural_net.output.logits
      ])
    else:
      ll = K.function(
          [neural_net.input, labels],
          [neural_net.output.log_prob(labels), neural_net.output.loc - labels])

    ensemble_metrics_vals = [
        ensemble_metrics(
            x_train, y_train, neural_net, ll, weight_files=ensemble_filenames),
        ensemble_metrics(
            x_test, y_test, neural_net, ll, weight_files=ensemble_filenames)
    ]


    for metrics, name in [(ensemble_metrics_vals, 'Ensemble model')]:
      print(name)
      for metrics_dict, split in [(metrics[0], 'Training'),
                                  (metrics[1], 'Testing')]:
        print(split)
        for metric_name in metrics_dict:
          print('{}:  {}'.format(metric_name, metrics_dict[metric_name]))

    try:
      tf.io.gfile.DeleteRecursively(model_dir)
    except tf.io.gfile.GOSError as oserror:
      print('GOSError: {0}'.format(oserror))


if __name__ == '__main__':
  app.run(main)
