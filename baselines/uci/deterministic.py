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

"""MLP on UCI data trained with maximum likelihood and gradient descent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import utils  # local file import
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

flags.DEFINE_enum('dataset', 'boston_housing',
                  enum_values=['boston_housing',
                               'concrete_strength',
                               'energy_efficiency',
                               'naval_propulsion',
                               'kin8nm',
                               'power_plant',
                               'protein_structure',
                               'wine',
                               'yacht_hydrodynamics'],
                  help='Name of the UCI dataset.')
flags.DEFINE_integer('ensemble_size', 1, 'Number of ensemble members.')
flags.DEFINE_boolean('bootstrap', False,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 2500, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('epsilon', 0.,
                   'Epsilon for adversarial training. It is given as a ratio '
                   'of the input range (e.g the adjustment is 2.55 if input '
                   'range is [0,255]). Set to 0. for no adversarial training.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/uci',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0,
                     'Random seed. Note train/test splits are random and also '
                     'based on this seed.')
FLAGS = flags.FLAGS


def multilayer_perceptron(input_shape, output_scaler=1.):
  """Builds a single hidden layer feedforward network.

  Args:
    input_shape: tf.TensorShape.
    output_scaler: Float to scale mean predictions. Training is faster and more
      stable when both the inputs and outputs are normalized. To not affect
      metrics such as RMSE and NLL, the outputs need to be scaled back
      (de-normalized, but the mean doesn't matter), using output_scaler.

  Returns:
    tf.keras.Model.
  """

  def output_fn(inputs):
    loc, untransformed_scale = inputs
    return ed.Normal(loc=loc, scale=tf.nn.softplus(untransformed_scale))

  inputs = tf.keras.layers.Input(shape=input_shape)
  hidden = tf.keras.layers.Dense(50, activation='relu')(inputs)
  loc = tf.keras.layers.Dense(1, activation=None)(hidden)
  untransformed_scale = tfp.layers.VariableLayer(shape=())(loc)
  outputs = tf.keras.layers.Lambda(output_fn)(
      (loc * output_scaler, untransformed_scale))
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def main(argv):
  del argv  # unused arg
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf1.disable_v2_behavior()

  x_train, y_train, x_test, y_test = utils.load(FLAGS.dataset)
  n_train = x_train.shape[0]

  session = tf1.Session()
  ensemble_filenames = []
  for i in range(FLAGS.ensemble_size):
    # TODO(trandustin): We re-build the graph for each ensemble member. This
    # is due to an unknown bug where the variables are otherwise not
    # re-initialized to be random. While this is inefficient in graph mode, I'm
    # keeping this for now as we'd like to move to eager mode anyways.
    model = multilayer_perceptron(
        x_train.shape[1:], np.std(y_train, axis=0) + tf.keras.backend.epsilon())

    def negative_log_likelihood(y, rv_y):
      del rv_y  # unused arg
      return -model.output.distribution.log_prob(y)  # pylint: disable=cell-var-from-loop

    def mse(y_true, y_sample):
      del y_sample  # unused arg
      return tf.math.square(model.output.distribution.loc - y_true)  # pylint: disable=cell-var-from-loop

    def log_likelihood(y_true, y_sample):
      del y_sample  # unused arg
      return model.output.distribution.log_prob(y_true)  # pylint: disable=cell-var-from-loop

    if FLAGS.epsilon:
      y_true = tf.keras.Input(shape=y_train.shape[1:], name='labels')
      loss = tf.reduce_mean(-model.output.distribution.log_prob(y_true))
      nn_input_tensor = model.input
      grad = tf1.gradients(loss, nn_input_tensor)[0]
      # It is assumed that the training data is normalized.
      adv_inputs_tensor = nn_input_tensor + FLAGS.epsilon * tf.math.sign(
          tf1.stop_gradient(grad))
      adv_inputs = tf.keras.Input(tensor=adv_inputs_tensor, name='adv_inputs')
      adv_out_dist = model(adv_inputs)
      adv_loss = tf.reduce_mean(-adv_out_dist.distribution.log_prob(y_true))
      optimizer = tf1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
      train_op = optimizer.minimize(0.5 * loss + 0.5 * adv_loss)
    else:
      model.compile(
          optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
          loss=negative_log_likelihood,
          metrics=[log_likelihood, mse])

    member_dir = os.path.join(FLAGS.output_dir, 'member_' + str(i))
    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=member_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)
    if FLAGS.epsilon:
      session.run(tf1.initialize_all_variables())
      for epoch in range((FLAGS.batch_size * FLAGS.training_steps) // n_train):
        logging.info('Epoch %s', epoch)
        for j in range(n_train // FLAGS.batch_size):
          perm = np.random.permutation(n_train)
          session.run(
              train_op,
              feed_dict={
                  nn_input_tensor: x_train[perm[j:j + FLAGS.batch_size]],
                  y_true: y_train[perm[j:j + FLAGS.batch_size]],
              })
    else:
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
          verbose=0,
          callbacks=[tensorboard])

    member_filename = os.path.join(member_dir, 'model.weights')
    ensemble_filenames.append(member_filename)
    model.save_weights(member_filename)

  labels = tf.keras.layers.Input(shape=y_train.shape[1:])
  ll = tf.keras.backend.function(
      [model.input, labels],
      [model.output.distribution.log_prob(labels),
       model.output.distribution.loc - labels])

  ensemble_metrics_vals = {
      'train': utils.ensemble_metrics(
          x_train, y_train, model, ll, weight_files=ensemble_filenames),
      'test': utils.ensemble_metrics(
          x_test, y_test, model, ll, weight_files=ensemble_filenames),
  }

  for split, metrics in ensemble_metrics_vals.items():
    logging.info(split)
    for metric_name in metrics:
      logging.info('%s: %s', metric_name, metrics[metric_name])

if __name__ == '__main__':
  app.run(main)
