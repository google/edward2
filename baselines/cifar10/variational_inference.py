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

"""Variational inference for ResNet-20 on CIFAR-10.

This script performs variational inference with a few notable techniques:

1. Normal prior whose mean is tied at the variational posterior's. This makes
   the KL penalty only penalize the weight posterior's standard deviation and
   not its mean. The prior's standard deviation is a fixed hyperparameter.
2. Fully factorized normal variational distribution (Blundell et al., 2015).
3. Flipout for lower-variance gradients in convolutional layers (Wen et al.,
   2018) but only applied to the second convolution in each residual block
   (Ovadia et al., 2019).
4. Variational dropout (local reparameterization) for lower-variance gradients
   in dense layers (Kingma et al., 2015).
5. Option to turn off batch normalization and use SELU activation and
   fixup initialization (Ovadia et al., 2019; Heek and Kalchbrenner, 2019).
   Batch normalization is enabled by default however as it performs better and
   is more stable.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from absl import flags
from absl import logging
import edward2 as ed
import utils  # local file import

from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('output_dir', '/tmp/cifar10', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_float('init_learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('prior_stddev', 0.1, 'Fixed stddev for weight prior.')
flags.DEFINE_boolean('batch_norm', True, 'Whether to apply batchnorm.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


class NormalKLDivergenceWithTiedMean(tf.keras.regularizers.Regularizer):
  """KL with normal prior whose mean is fixed at the variational posterior's."""

  def __init__(self, stddev=1., scale_factor=1.):
    """Constructs regularizer."""
    self.stddev = stddev
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an ed.Normal random variable as input."""
    if not isinstance(x, ed.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    prior = ed.Independent(
        ed.Normal(loc=x.distribution.mean(), scale=self.stddev).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    regularization = x.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'stddev': self.stddev,
        'scale_factor': self.scale_factor,
    }


def resnet_layer(inputs,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 depth=20,
                 batch_norm=True,
                 bayesian=False,
                 prior_stddev=1.,
                 dataset_size=None):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    kernel_size: Kernel dimensions for Conv2D.
    strides: Stride dimensinons for Conv2D.
    activation: tf.keras.activations.Activation.
    depth: ResNet depth.
    batch_norm: Whether to apply batch normalization.
    bayesian: Whether to apply Bayesian layers.
    prior_stddev: Standard deviation of weight priors.
    dataset_size: Total number of examples in an epoch.

  Returns:
    tf.Tensor.
  """
  if bayesian:
    def fixup_init(shape, dtype=None):
      """Simplified form of fixup initialization (Zhang et al., 2019)."""
      return (tf.keras.initializers.he_normal()(shape, dtype=dtype) *
              depth**(-1/4))
    if batch_norm:
      kernel_initializer = 'trainable_he_normal'
    else:
      kernel_initializer = ed.initializers.TrainableNormal(
          mean_initializer=fixup_init)
    conv = ed.layers.Conv2DFlipout(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=NormalKLDivergenceWithTiedMean(
            stddev=prior_stddev, scale_factor=1./dataset_size))
  else:
    conv = tf.keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4))

  x = inputs
  logging.info('Applying conv layer.')
  x = conv(x)
  if batch_norm:
    x = tf.keras.layers.BatchNormalization()(x)
  if activation is not None:
    x = tf.keras.layers.Activation(activation)(x)
  return x


def resnet_v1(input_shape,
              depth,
              num_classes,
              batch_norm,
              prior_stddev,
              dataset_size):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    batch_norm: Whether to apply batch normalization.
    prior_stddev: Standard deviation of weight priors.
    dataset_size: Total number of examples in an epoch.

  Returns:
    tf.keras.Model.
  """
  num_res_blocks = (depth - 2) // 6
  filters = 16
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

  layer = functools.partial(resnet_layer,
                            depth=depth,
                            dataset_size=dataset_size,
                            prior_stddev=prior_stddev)
  activation = 'relu' if batch_norm else 'selu'

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = layer(inputs,
            filters=filters,
            activation=activation)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = layer(x,
                filters=filters,
                strides=strides,
                activation=activation,
                batch_norm=batch_norm)
      y = layer(y,
                filters=filters,
                activation=None,
                batch_norm=batch_norm,
                bayesian=True)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = layer(x,
                  filters=filters,
                  kernel_size=1,
                  strides=strides,
                  activation=None,
                  batch_norm=False)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.Activation(activation)(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  # TODO(trandustin): Change to DenseVariationalDropout; doesn't work with v2.
  x = ed.layers.DenseFlipout(
      num_classes,
      kernel_initializer='trainable_he_normal',
      kernel_regularizer=NormalKLDivergenceWithTiedMean(
          stddev=prior_stddev,
          scale_factor=1./dataset_size))(x)
  return tf.keras.models.Model(inputs=inputs, outputs=x)


def get_metrics(model, dataset_size):
  """Get metrics for the model."""

  def kl(y_true, y_pred):
    """KL divergence."""
    del y_true, y_pred  # unused arg
    return sum(model.losses) * dataset_size

  def elbo(y_true, y_pred):
    """Evidence lower bound."""
    y_true = tf.squeeze(tf.cast(y_true, tf.int32))
    log_likelihood = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true)
    log_likelihood *= dataset_size
    return log_likelihood - kl(y_true, y_pred)

  return kl, elbo


def main(argv):
  del argv  # unused arg
  if FLAGS.num_cores > 1 or not FLAGS.use_gpu:
    raise ValueError('Only single GPU is currently supported.')
  tf.enable_v2_behavior()
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  dataset_train, ds_info = utils.load_dataset(tfds.Split.TRAIN, with_info=True)
  dataset_size = ds_info.splits['train'].num_examples
  dataset_train = dataset_train.repeat().shuffle(10 * FLAGS.batch_size).batch(
      FLAGS.batch_size)
  test_batch_size = 100
  validation_steps = ds_info.splits['test'].num_examples // test_batch_size
  dataset_test = utils.load_dataset(tfds.Split.TEST)
  dataset_test = dataset_test.repeat().batch(test_batch_size)

  model = resnet_v1(input_shape=ds_info.features['image'].shape,
                    depth=20,
                    num_classes=ds_info.features['label'].num_classes,
                    batch_norm=FLAGS.batch_norm,
                    prior_stddev=FLAGS.prior_stddev,
                    dataset_size=dataset_size)
  kl, elbo = get_metrics(model, dataset_size)

  model.compile(
      tf.keras.optimizers.Adam(FLAGS.init_learning_rate),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          tf.keras.metrics.SparseCategoricalCrossentropy(
              name='negative_log_likelihood',
              from_logits=True),
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
          elbo,
          kl])
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.output_dir,
                                                  write_graph=False)
  lr_scheduler = utils.make_lr_scheduler(FLAGS.init_learning_rate)
  model.fit(dataset_train,
            steps_per_epoch=dataset_size // FLAGS.batch_size,
            epochs=FLAGS.train_epochs,
            validation_data=dataset_test,
            validation_steps=validation_steps,
            callbacks=[tensorboard_cb, lr_scheduler])

  logging.info('Saving model to output_dir.')
  model_filename = FLAGS.output_dir + '/model.ckpt'
  model.save_weights(model_filename)

if __name__ == '__main__':
  app.run(main)
