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

"""Variational inference for MLP on UCI data."""

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
flags.DEFINE_integer('training_steps', 30000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('learning_rate_for_sampling', 0.00001, 'Learning rate.')
flags.DEFINE_integer('auxiliary_sampling_frequency', 100,
                     'Steps between sampling auxiliary variables.')
flags.DEFINE_float('auxiliary_variance_ratio', 0.7,
                   'Variance ratio of the auxiliary variables wrt the prior.')
flags.DEFINE_integer('n_auxiliary_variables', 5,
                     'Number of auxiliary variables.')
flags.DEFINE_integer('ensemble_size', 10, 'Number of ensemble components.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/uci',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
FLAGS = flags.FLAGS


def multilayer_perceptron(n_examples, input_shape, output_scaler=1.):
  """Builds a single hidden layer Bayesian feedforward network.

  Args:
    n_examples: Number of examples in training set.
    input_shape: tf.TensorShape.
    output_scaler: Float to scale mean predictions. Training is faster and more
      stable when both the inputs and outputs are normalized. To not affect
      metrics such as RMSE and NLL, the outputs need to be scaled back
      (de-normalized, but the mean doesn't matter), using output_scaler.

  Returns:
    tf.keras.Model.
  """
  p_fn, q_fn = utils.mean_field_fn(empirical_bayes=True)
  def normalized_kl_fn(q, p, _):
    return q.kl_divergence(p) / tf.cast(n_examples, tf.float32)

  inputs = tf.keras.layers.Input(shape=input_shape)
  hidden = tfp.layers.DenseLocalReparameterization(
      50,
      activation='relu',
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(inputs)
  loc = tfp.layers.DenseLocalReparameterization(
      1,
      activation=None,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(hidden)
  loc = tf.keras.layers.Lambda(lambda x: x * output_scaler)(loc)
  scale = tfp.layers.VariableLayer(
      shape=(), initializer=tf.keras.initializers.Constant(-3.))(loc)
  scale = tf.keras.layers.Activation('softplus')(scale)
  outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(loc=x[0], scale=x[1]))(
      (loc, scale))
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_losses_and_metrics(model, n_train):
  """Define the losses and metrics for the model."""

  def negative_log_likelihood(y, rv_y):
    del rv_y  # unused arg
    return -model.output.distribution.log_prob(y)

  def mse(y_true, y_sample):
    """Mean-squared error."""
    del y_sample  # unused arg
    return tf.math.square(model.output.distribution.loc - y_true)

  def log_likelihood(y_true, y_sample):
    del y_sample  # unused arg
    return model.output.distribution.log_prob(y_true)

  def kl(y_true, y_sample):
    """KL-divergence."""
    del y_true  # unused arg
    del y_sample  # unused arg
    sampling_cost = sum(
        [l.kl_cost_weight + l.kl_cost_bias for l in model.layers])
    return sum(model.losses) * n_train + sampling_cost

  def elbo(y_true, y_sample):
    return log_likelihood(y_true, y_sample) * n_train - kl(y_true, y_sample)

  return negative_log_likelihood, mse, log_likelihood, kl, elbo


def main(argv):
  del argv  # unused arg
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf1.disable_v2_behavior()

  session = tf1.Session()
  with session.as_default():
    x_train, y_train, x_test, y_test = utils.load(FLAGS.dataset)
    n_train = x_train.shape[0]

    model = multilayer_perceptron(
        n_train,
        x_train.shape[1:],
        np.std(y_train) + tf.keras.backend.epsilon())
    for l in model.layers:
      l.kl_cost_weight = l.add_weight(
          name='kl_cost_weight',
          shape=(),
          initializer=tf.constant_initializer(0.),
          trainable=False)
      l.kl_cost_bias = l.add_variable(
          name='kl_cost_bias',
          shape=(),
          initializer=tf.constant_initializer(0.),
          trainable=False)

    [negative_log_likelihood,
     mse,
     log_likelihood,
     kl,
     elbo] = get_losses_and_metrics(model, n_train)
    metrics = [elbo, log_likelihood, kl, mse]

    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=FLAGS.output_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)

    def fit_fn(model,
               steps,
               initial_epoch):
      return model.fit(
          x=x_train,
          y=y_train,
          batch_size=FLAGS.batch_size,
          epochs=initial_epoch + (FLAGS.batch_size * steps) // n_train,
          initial_epoch=initial_epoch,
          validation_data=(x_test, y_test),
          validation_freq=max(
              (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
          verbose=1,
          callbacks=[tensorboard])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=float(FLAGS.learning_rate)),
        loss=negative_log_likelihood,
        metrics=metrics)
    session.run(tf1.initialize_all_variables())

    train_epochs = (FLAGS.training_steps * FLAGS.batch_size) // n_train
    fit_fn(model, FLAGS.training_steps, initial_epoch=0)

    labels = tf.keras.layers.Input(shape=y_train.shape[1:])
    ll = tf.keras.backend.function(
        [model.input, labels],
        [model.output.distribution.log_prob(labels),
         model.output.distribution.loc - labels])

    base_metrics = [
        utils.ensemble_metrics(x_train, y_train, model, ll),
        utils.ensemble_metrics(x_test, y_test, model, ll),
    ]
    model_dir = os.path.join(FLAGS.output_dir, 'models')
    tf.io.gfile.makedirs(model_dir)
    base_model_filename = os.path.join(model_dir, 'base_model.weights')
    model.save_weights(base_model_filename)

    # Train base model further for comparison.
    fit_fn(
        model,
        FLAGS.n_auxiliary_variables * FLAGS.auxiliary_sampling_frequency *
        FLAGS.ensemble_size,
        initial_epoch=train_epochs)

    overtrained_metrics = [
        utils.ensemble_metrics(x_train, y_train, model, ll),
        utils.ensemble_metrics(x_test, y_test, model, ll),
    ]

    # Perform refined VI.
    sample_op = []
    for l in model.layers:
      if hasattr(l, 'kernel_prior'):
        weight_op, weight_cost = utils.sample_auxiliary_op(
            l.kernel_prior.distribution, l.kernel_posterior.distribution,
            FLAGS.auxiliary_variance_ratio)
        sample_op.append(weight_op)
        sample_op.append(l.kl_cost_weight.assign_add(weight_cost))
        # Fix the variance of the prior
        session.run(l.kernel_prior.distribution.istrainable.assign(0.))
        if hasattr(l.bias_prior, 'distribution'):
          bias_op, bias_cost = utils.sample_auxiliary_op(
              l.bias_prior.distribution, l.bias_posterior.distribution,
              FLAGS.auxiliary_variance_ratio)
          sample_op.append(bias_op)
          sample_op.append(l.kl_cost_bias.assign_add(bias_cost))
          # Fix the variance of the prior
          session.run(l.bias_prior.distribution.istrainable.assign(0.))

    ensemble_filenames = []
    for i in range(FLAGS.ensemble_size):
      model.load_weights(base_model_filename)
      for j in range(FLAGS.n_auxiliary_variables):
        session.run(sample_op)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                # The learning rate is proportional to the scale of the prior.
                lr=float(FLAGS.learning_rate_for_sampling *
                         np.sqrt(1. - FLAGS.auxiliary_variance_ratio)**j)),
            loss=negative_log_likelihood,
            metrics=metrics)
        fit_fn(
            model,
            FLAGS.auxiliary_sampling_frequency,
            initial_epoch=train_epochs)
      ensemble_filename = os.path.join(
          model_dir, 'ensemble_component_' + str(i) + '.weights')
      ensemble_filenames.append(ensemble_filename)
      model.save_weights(ensemble_filename)

    auxiliary_metrics = [
        utils.ensemble_metrics(
            x_train,
            y_train,
            model,
            ll,
            weight_files=ensemble_filenames),
        utils.ensemble_metrics(
            x_test,
            y_test,
            model,
            ll,
            weight_files=ensemble_filenames),
    ]

    for metrics, name in [(base_metrics, 'Base model'),
                          (overtrained_metrics, 'Overtrained model'),
                          (auxiliary_metrics, 'Auxiliary sampling')]:
      logging.info(name)
      for metrics_dict, split in [(metrics[0], 'train'),
                                  (metrics[1], 'test')]:
        logging.info(split)
        for metric_name in metrics_dict:
          logging.info('%s: %s', metric_name, metrics_dict[metric_name])


if __name__ == '__main__':
  app.run(main)
