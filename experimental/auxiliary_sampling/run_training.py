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

"""Variational inference for LeNet5 or ResNet-20 on CIFAR-10."""

import os
from absl import app
from absl import flags
from absl import logging

from experimental.auxiliary_sampling import datasets  # local file import
from experimental.auxiliary_sampling.compute_metrics import ensemble_metrics  # local file import
from experimental.auxiliary_sampling.lenet5 import lenet5  # local file import
from experimental.auxiliary_sampling.res_net import res_net  # local file import
from experimental.auxiliary_sampling.sampling import sample_auxiliary_op  # local file import
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

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
flags.DEFINE_boolean('resnet', False, 'Use a ResNet for image classification.' +
                     'The default is to use the LeNet5 arhitecture.' +
                     'Currently only supported on cifar10.')
flags.DEFINE_boolean('hybrid', False, 'Use mix of deterministic and Bayesian.' +
                     'Only applies when resnet is True.')
flags.DEFINE_boolean('batchnorm', False,
                     'Use batchnorm. Only applies when resnet is True.')
flags.DEFINE_boolean(
    'data_augmentation', False,
    'Use data augmentation. Only applies when resnet is True.')
FLAGS = flags.FLAGS


def get_losses_and_metrics(model, n_train):
  """Define the losses and metrics for the model."""

  def negative_log_likelihood(y, rv_y):
    del rv_y  # unused arg
    return -model.output.distribution.log_prob(tf.squeeze(y))

  def accuracy(y_true, y_sample):
    del y_sample  # unused arg
    return tf.equal(
        tf.argmax(input=model.output.distribution.logits, axis=1),
        tf.cast(tf.squeeze(y_true), tf.int64))

  def log_likelihood(y_true, y_sample):
    """Expected conditional log-likelihood."""
    del y_sample  # unused arg
    return model.output.distribution.log_prob(tf.squeeze(y_true))

  def kl(y_true, y_sample):
    """KL-divergence."""
    del y_true  # unused arg
    del y_sample  # unused arg
    sampling_cost = sum(
        [l.kl_cost_weight + l.kl_cost_bias for l in model.layers])
    return sum(model.losses) * n_train + sampling_cost

  def elbo(y_true, y_sample):
    return log_likelihood(y_true, y_sample) * n_train - kl(y_true, y_sample)

  return negative_log_likelihood, accuracy, log_likelihood, kl, elbo


def main(argv):
  del argv  # unused arg
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf1.disable_v2_behavior()

  session = tf1.Session()
  with session.as_default():
    x_train, y_train, x_test, y_test = datasets.load(session)
    n_train = x_train.shape[0]

    num_classes = int(np.amax(y_train)) + 1
    if not FLAGS.resnet:
      model = lenet5(n_train, x_train.shape[1:], num_classes)
    else:
      datagen = tf.python.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=90,
          width_shift_range=0.1,
          height_shift_range=0.1,
          horizontal_flip=True)
      datagen.fit(x_train)
      model = res_net(n_train,
                      x_train.shape[1:],
                      num_classes,
                      batchnorm=FLAGS.batchnorm,
                      variational='hybrid' if FLAGS.hybrid else 'full')

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

      lr_callback = tf.python.keras.callbacks.LearningRateScheduler(schedule_fn)

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
     accuracy,
     log_likelihood,
     kl,
     elbo] = get_losses_and_metrics(model, n_train)

    metrics = [elbo, log_likelihood, kl, accuracy]

    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=FLAGS.output_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)
    if FLAGS.resnet:
      callbacks = [tensorboard, lr_callback]
    else:
      callbacks = [tensorboard]

    if not FLAGS.resnet or not FLAGS.data_augmentation:

      def fit_fn(model,
                 steps,
                 initial_epoch=0,
                 with_lr_schedule=FLAGS.resnet):
        return model.fit(
            x=x_train,
            y=y_train,
            batch_size=FLAGS.batch_size,
            epochs=initial_epoch + (FLAGS.batch_size * steps) // n_train,
            initial_epoch=initial_epoch,
            validation_data=(x_test, y_test),
            validation_freq=(
                (FLAGS.validation_freq * FLAGS.batch_size) // n_train),
            verbose=1,
            callbacks=callbacks if with_lr_schedule else [tensorboard])
    else:

      def fit_fn(model,
                 steps,
                 initial_epoch=0,
                 with_lr_schedule=FLAGS.resnet):
        return model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
            epochs=initial_epoch + (FLAGS.batch_size * steps) // n_train,
            initial_epoch=initial_epoch,
            steps_per_epoch=n_train // FLAGS.batch_size,
            validation_data=(x_test, y_test),
            validation_freq=max(
                (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
            verbose=1,
            callbacks=callbacks if with_lr_schedule else [tensorboard])

    model.compile(
        optimizer=tf.python.keras.optimizers.Adam(lr=float(FLAGS.learning_rate)),
        loss=negative_log_likelihood,
        metrics=metrics)
    session.run(tf1.initialize_all_variables())

    train_epochs = (FLAGS.training_steps * FLAGS.batch_size) // n_train
    fit_fn(model, FLAGS.training_steps)

    labels = tf.python.keras.layers.Input(shape=y_train.shape[1:])
    ll = tf.python.keras.backend.function([model.input, labels], [
        model.output.distribution.log_prob(tf.squeeze(labels)),
        model.output.distribution.logits
    ])

    base_metrics = [
        ensemble_metrics(x_train, y_train, model, ll),
        ensemble_metrics(x_test, y_test, model, ll)
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
        ensemble_metrics(x_train, y_train, model, ll),
        ensemble_metrics(x_test, y_test, model, ll)
    ]

    # Perform refined VI.
    sample_op = []
    for l in model.layers:
      if isinstance(l, tfp.layers.DenseLocalReparameterization) or isinstance(
          l, tfp.layers.Convolution2DFlipout):
        weight_op, weight_cost = sample_auxiliary_op(
            l.kernel_prior.distribution, l.kernel_posterior.distribution,
            FLAGS.auxiliary_variance_ratio)
        sample_op.append(weight_op)
        sample_op.append(l.kl_cost_weight.assign_add(weight_cost))
        # Fix the variance of the prior
        session.run(l.kernel_prior.distribution.istrainable.assign(0.))
        if hasattr(l.bias_prior, 'distribution'):
          bias_op, bias_cost = sample_auxiliary_op(
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
            optimizer=tf.python.keras.optimizers.Adam(
                # The learning rate is proportional to the scale of the prior.
                lr=float(FLAGS.learning_rate_for_sampling *
                         np.sqrt(1. - FLAGS.auxiliary_variance_ratio)**j)),
            loss=negative_log_likelihood,
            metrics=metrics)
        fit_fn(
            model,
            FLAGS.auxiliary_sampling_frequency,
            initial_epoch=train_epochs,
            with_lr_schedule=False)
      ensemble_filename = os.path.join(
          model_dir, 'ensemble_component_' + str(i) + '.weights')
      ensemble_filenames.append(ensemble_filename)
      model.save_weights(ensemble_filename)

    auxiliary_metrics = [
        ensemble_metrics(
            x_train,
            y_train,
            model,
            ll,
            weight_files=ensemble_filenames),
        ensemble_metrics(
            x_test,
            y_test,
            model,
            ll,
            weight_files=ensemble_filenames)
    ]

    for metrics, name in [(base_metrics, 'Base model'),
                          (overtrained_metrics, 'Overtrained model'),
                          (auxiliary_metrics, 'Auxiliary sampling')]:
      logging.info(name)
      for metrics_dict, split in [(metrics[0], 'Training'),
                                  (metrics[1], 'Testing')]:
        logging.info(split)
        for metric_name in metrics_dict:
          logging.info('%s: %s', metric_name, metrics_dict[metric_name])


if __name__ == '__main__':
  app.run(main)
