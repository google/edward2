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

"""Train a BNN on a UCI or image dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import app
from absl import flags

from edward2.experimental.auxiliary_sampling import datasets
from edward2.experimental.auxiliary_sampling.compute_metrics import ensemble_metrics
from edward2.experimental.auxiliary_sampling.conv_net import conv_net
from edward2.experimental.auxiliary_sampling.res_net import res_net
from edward2.experimental.auxiliary_sampling.sampling import sample_auxiliary_op
from edward2.experimental.auxiliary_sampling.simple_net import simple_net
from keras import backend as K
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'Name of the UCI or image dataset.')
flags.DEFINE_integer('training_steps', 15000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('learning_rate_for_sampling', 0.00001, 'Learning rate.')
flags.DEFINE_integer('auxiliary_sampling_frequency', 100,
                     'Steps between sampling auxiliary variables.')
flags.DEFINE_float('auxiliary_variance_ratio', 0.7,
                   'Variance ratio of the auxiliary variables wrt the prior.')
flags.DEFINE_integer('n_auxiliary_variables', 5,
                     'Number of auxiliary variables.')
flags.DEFINE_integer('n_ensemble', 10, 'Number of ensemble components.')
flags.DEFINE_integer('measurement_frequency', 200, 'Measurement frequency.')
flags.DEFINE_string('working_dir', '/tmp', 'Working directory.')
flags.DEFINE_integer('seed', None, 'Random seed.')
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


def get_losses_and_metrics(neural_net, n_train, classification):
  """Define the losses and metrics for the model."""

  def negative_log_likelihood(y, rv_y):
    del rv_y  # unused arg
    return -neural_net.output.log_prob(
        tf.squeeze(y)) if classification else -neural_net.output.log_prob(y)

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
    return neural_net.output.log_prob(tf.squeeze(
        y_true)) if classification else neural_net.output.log_prob(y_true)

  def kl(y_true, y_sample):
    """KL-divergence."""
    del y_true  # unused arg
    del y_sample  # unused arg
    sampling_cost = sum(
        [l.kl_cost_weight + l.kl_cost_bias for l in neural_net.layers])
    return sum(neural_net.losses) * n_train + sampling_cost

  def elbo(y_true, y_sample):
    return mll(y_true, y_sample) * n_train - kl(y_true, y_sample)

  return negative_log_likelihood, accuracy, mse, mll, kl, elbo


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

    x_train, y_train, x_test, y_test = datasets.load(FLAGS.dataset, session)
    n_train = x_train.shape[0]

    # Build net
    if image_classification:
      num_classes = int(np.amax(y_train)) + 1
      if not FLAGS.resnet:
        # The default is a LeNet-5 architecture.
        neural_net = conv_net(n_train, x_train.shape[1:], num_classes)
      else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=90,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        datagen.fit(x_train)
        neural_net = res_net(n_train,
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

        lr_callback = tf.keras.callbacks.LearningRateScheduler(schedule_fn)
    else:
      # The output of the network is scaled by y_std to improve stability.
      neural_net = simple_net(n_train, x_train.shape[1:],
                              np.std(y_train) + 1e-10)
    for l in neural_net.layers:
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

    negative_log_likelihood, accuracy, mse, mll, kl, elbo = get_losses_and_metrics(
        neural_net, n_train, image_classification)

    if image_classification:
      metrics = [elbo, mll, kl, accuracy]
    else:
      metrics = [elbo, mll, kl, mse]

    # Training
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.working_dir,
        update_freq=FLAGS.batch_size * FLAGS.measurement_frequency)
    if FLAGS.resnet:
      callbacks = [tensorboard, lr_callback]
    else:
      callbacks = [tensorboard]

    if not FLAGS.resnet or not FLAGS.data_augmentation:

      def fit_fn(neural_net,
                 iterations,
                 initial_epoch=0,
                 with_lr_schedule=FLAGS.resnet):
        return neural_net.fit(
            x=x_train,
            y=y_train,
            batch_size=FLAGS.batch_size,
            epochs=initial_epoch +
            math.ceil(FLAGS.batch_size * iterations / n_train),
            initial_epoch=initial_epoch,
            validation_data=(x_test, y_test),
            validation_freq=math.ceil(FLAGS.measurement_frequency *
                                      FLAGS.batch_size / n_train),
            verbose=1,
            callbacks=callbacks if with_lr_schedule else [tensorboard])
    else:

      def fit_fn(neural_net,
                 iterations,
                 initial_epoch=0,
                 with_lr_schedule=FLAGS.resnet):
        return neural_net.fit_generator(
            datagen.flow(x_train, y_train, batch_size=FLAGS.batch_size),
            epochs=initial_epoch +
            math.ceil(FLAGS.batch_size * iterations / n_train),
            initial_epoch=initial_epoch,
            steps_per_epoch=math.ceil(n_train / FLAGS.batch_size),
            validation_data=(x_test, y_test),
            validation_freq=math.ceil(FLAGS.measurement_frequency *
                                      FLAGS.batch_size / n_train),
            verbose=1,
            callbacks=callbacks if with_lr_schedule else [tensorboard])

    neural_net.compile(
        optimizer=tf.keras.optimizers.Adam(lr=float(FLAGS.learning_rate)),
        loss=negative_log_likelihood,
        metrics=metrics)
    session.run(tf.initialize_all_variables())

    # Train base model
    train_epochs = math.ceil(FLAGS.training_steps * FLAGS.batch_size / n_train)
    fit_fn(neural_net, FLAGS.training_steps)

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

    base_metrics = [
        ensemble_metrics(x_train, y_train, neural_net, ll),
        ensemble_metrics(x_test, y_test, neural_net, ll)
    ]
    model_dir = os.path.join(FLAGS.working_dir, 'models')
    tf.io.gfile.MakeDirs(model_dir)
    base_model_filename = os.path.join(model_dir, 'base_model.weights')
    neural_net.save_weights(base_model_filename)

    # Train base model further for comparison
    fit_fn(
        neural_net,
        FLAGS.n_auxiliary_variables * FLAGS.auxiliary_sampling_frequency *
        FLAGS.n_ensemble,
        initial_epoch=train_epochs)

    overtrained_metrics = [
        ensemble_metrics(x_train, y_train, neural_net, ll),
        ensemble_metrics(x_test, y_test, neural_net, ll)
    ]

    # Sampling operation
    sample_op = []
    for l in neural_net.layers:
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

    # Ensemble training
    ensemble_filenames = []
    for i in range(FLAGS.n_ensemble):
      neural_net.load_weights(base_model_filename)
      for j in range(FLAGS.n_auxiliary_variables):
        session.run(sample_op)
        neural_net.compile(
            optimizer=tf.keras.optimizers.Adam(
                # The learning rate is proportional to the scale of the prior.
                lr=float(FLAGS.learning_rate_for_sampling *
                         np.sqrt(1. - FLAGS.auxiliary_variance_ratio)**j)),
            loss=negative_log_likelihood,
            metrics=metrics)
        fit_fn(
            neural_net,
            FLAGS.auxiliary_sampling_frequency,
            initial_epoch=train_epochs,
            with_lr_schedule=False)
      ensemble_filename = os.path.join(
          model_dir, 'ensemble_component_' + str(i) + '.weights')
      ensemble_filenames.append(ensemble_filename)
      neural_net.save_weights(ensemble_filename)

    auxiliary_metrics = [
        ensemble_metrics(
            x_train,
            y_train,
            neural_net,
            ll,
            weight_files=ensemble_filenames),
        ensemble_metrics(
            x_test,
            y_test,
            neural_net,
            ll,
            weight_files=ensemble_filenames)
    ]


    # Print metrics
    for metrics, name in [(base_metrics, 'Base model'),
                          (overtrained_metrics, 'Overtrained model'),
                          (auxiliary_metrics, 'Auxiliary sampling')]:
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
