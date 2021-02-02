# coding=utf-8
# Copyright 2021 The Edward2 Authors.
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

"""DNN on CIFAR-10 trained with maximum likelihood and gradient descent."""

import os

from absl import app
from absl import flags
from absl import logging

from experimental.auxiliary_sampling import datasets  # local file import
from experimental.auxiliary_sampling.compute_metrics import ensemble_metrics  # local file import
from experimental.auxiliary_sampling.deterministic_baseline.lenet5 import lenet5  # local file import
from experimental.auxiliary_sampling.res_net import res_net  # local file import
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

flags.DEFINE_integer('ensemble_size', 1, 'Number of ensemble members.')
flags.DEFINE_boolean('bootstrap', False,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 40000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/det_training',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_boolean(
    'resnet', False, 'Use a ResNet for image classification.' +
    'The default is to use the LeNet5 arhitecture.' +
    'Currently only supported on cifar10.')
flags.DEFINE_boolean('batchnorm', False,
                     'Use batchnorm. Only applies when resnet is True.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf1.disable_v2_behavior()

  session = tf1.Session()
  x_train, y_train, x_test, y_test = datasets.load(session)
  n_train = x_train.shape[0]
  num_classes = int(np.amax(y_train)) + 1

  ensemble_filenames = []
  for i in range(FLAGS.ensemble_size):
    # TODO(trandustin): We re-build the graph for each ensemble member. This
    # is due to an unknown bug where the variables are otherwise not
    # re-initialized to be random. While this is inefficient in graph mode, I'm
    # keeping this for now as we'd like to move to eager mode anyways.
    if not FLAGS.resnet:
      model = lenet5(x_train.shape[1:], num_classes)
    else:
      model = res_net(
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
        return rate

      lr_callback = tf.python.keras.callbacks.LearningRateScheduler(schedule_fn)

    def negative_log_likelihood(y, rv_y):
      del rv_y  # unused arg
      return -model.output.distribution.log_prob(tf.squeeze(y))  # pylint: disable=cell-var-from-loop

    def accuracy(y_true, y_sample):
      del y_sample  # unused arg
      return tf.equal(
          tf.argmax(input=model.output.distribution.logits, axis=1),  # pylint: disable=cell-var-from-loop
          tf.cast(tf.squeeze(y_true), tf.int64))

    def log_likelihood(y_true, y_sample):
      del y_sample  # unused arg
      return model.output.distribution.log_prob(tf.squeeze(y_true))  # pylint: disable=cell-var-from-loop

    model.compile(
        optimizer=tf.python.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss=negative_log_likelihood,
        metrics=[log_likelihood, accuracy])
    member_dir = os.path.join(FLAGS.output_dir, 'member_' + str(i))
    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=member_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)

    if FLAGS.bootstrap:
      inds = np.random.choice(n_train, n_train, replace=True)
      x_sampled = x_train[inds]
      y_sampled = y_train[inds]

    model.fit(
        x=x_train if not FLAGS.bootstrap else x_sampled,
        y=y_train if not FLAGS.bootstrap else y_sampled,
        batch_size=FLAGS.batch_size,
        epochs=(FLAGS.batch_size * FLAGS.training_steps) // n_train,
        validation_data=(x_test, y_test),
        validation_freq=max(
            (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
        verbose=1,
        callbacks=[tensorboard]
        if not FLAGS.resnet else [tensorboard, lr_callback])

    member_filename = os.path.join(member_dir, 'model.weights')
    ensemble_filenames.append(member_filename)
    model.save_weights(member_filename)

  labels = tf.python.keras.layers.Input(shape=y_train.shape[1:])
  ll = tf.python.keras.backend.function([model.input, labels], [
      model.output.distribution.log_prob(tf.squeeze(labels)),
      model.output.distribution.logits,
  ])

  ensemble_metrics_vals = {
      'train': ensemble_metrics(
          x_train, y_train, model, ll, weight_files=ensemble_filenames),
      'test': ensemble_metrics(
          x_test, y_test, model, ll, weight_files=ensemble_filenames),
  }

  for split, metrics in ensemble_metrics_vals.items():
    logging.info(split)
    for metric_name in metrics:
      logging.info('%s: %s', metric_name, metrics[metric_name])

if __name__ == '__main__':
  app.run(main)
