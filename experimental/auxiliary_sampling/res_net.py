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

"""Build a Convolutional Bayesian neural network."""

import functools

from absl import logging
import edward2 as ed
from experimental.auxiliary_sampling.sampling import mean_field_fn  # local file import

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

keras = tf.python.keras


def _resnet_layer(inputs,
                  num_filters=16,
                  kernel_size=3,
                  strides=1,
                  activation='relu',
                  depth=20,
                  batchnorm=False,
                  conv_first=True,
                  variational=False,
                  n_examples=None):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs (tensor): input tensor from input image or previous layer
    num_filters (int): Conv2D number of filters
    kernel_size (int): Conv2D square kernel dimensions
    strides (int): Conv2D square stride dimensions
    activation (string): Activation function string.
    depth (int): ResNet depth; used for initialization scale.
    batchnorm (bool): whether to include batch normalization
    conv_first (bool): conv-bn-activation (True) or bn-activation-conv (False)
    variational (bool): Whether to use a variational convolutional layer.
    n_examples (int): Number of examples per epoch for variational KL.

  Returns:
      x (tensor): tensor as input to the next layer
  """
  if variational:

    def fixup_init(shape, dtype=None):
      """Fixup initialization; see https://arxiv.org/abs/1901.09321."""
      return keras.initializers.he_normal()(
          shape, dtype=dtype) * depth**(-1 / 4)

    p_fn, q_fn = mean_field_fn(empirical_bayes=True, initializer=fixup_init)

    def normalized_kl_fn(q, p, _):
      return tfp.distributions.kl_divergence(q, p) / tf.to_float(n_examples)

    conv = tfp.layers.Convolution2DFlipout(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_prior_fn=p_fn,
        kernel_posterior_fn=q_fn,
        kernel_divergence_fn=normalized_kl_fn)
  else:
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=keras.regularizers.l2(1e-4))

  def apply_conv(net):
    return conv(net)

  x = inputs
  x = apply_conv(x) if conv_first else x
  if batchnorm:
    x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Activation(activation)(x) if activation is not None else x
  x = x if conv_first else apply_conv(x)
  return x


def build_resnet_v1(input_layer,
                    depth,
                    variational,
                    batchnorm,
                    n_examples):
  """ResNet Version 1 Model builder.

  Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
  Last ReLU is after the shortcut connection.
  At the beginning of each stage, the feature map size is halved (downsampled)
  by a convolutional layer with strides=2, while the number of filters is
  doubled. Within each stage, the layers have the same number filters and the
  same number of filters.
  Features maps sizes:
  stage 0: 32x32, 16
  stage 1: 16x16, 32
  stage 2:  8x8,  64
  The Number of parameters is approx:
  ResNet20 0.27M
  ResNet32 0.46M
  ResNet44 0.66M
  ResNet56 0.85M
  ResNet110 1.7M

  Args:
    input_layer (tensor): keras.layers.InputLayer instance.
    depth (int): number of core convolutional layers. It should be 6n+2 (eg 20,
      32, 44).
    variational (str): 'none', 'hybrid', 'full'. whether to use variational
      inference for zero, some, or all layers.
    batchnorm (bool): use of batchnorm layers.
    n_examples (int): number of training points.

  Returns:
     output: the output tensor of the Network.
  """
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
  # Start model definition.
  num_filters = 16
  num_res_blocks = int((depth - 2) / 6)

  activation = 'selu' if variational else 'relu'
  resnet_layer = functools.partial(
      _resnet_layer,
      depth=depth,
      n_examples=n_examples)

  logging.info('Starting ResNet build.')
  x = resnet_layer(inputs=input_layer,
                   activation=activation)
  # Instantiate the stack of residual units
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(
          inputs=x,
          num_filters=num_filters,
          strides=strides,
          activation=activation,
          variational=True if variational == 'full' else False,
          batchnorm=batchnorm)
      y = resnet_layer(
          inputs=y,
          num_filters=num_filters,
          activation=None,
          variational=True if variational in ('hybrid', 'full') else False,
          batchnorm=batchnorm)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(
            inputs=x,
            num_filters=num_filters,
            kernel_size=1,
            strides=strides,
            activation=None,
            batchnorm=False)
      x = keras.layers.add([x, y])
      x = keras.layers.Activation(activation)(x)
    num_filters *= 2

  # Add classifier on top.
  # v1 does not use BN after last shortcut connection-ReLU
  x = keras.layers.AveragePooling2D(pool_size=8)(x)
  return keras.layers.Flatten()(x)


def res_net(n_examples,
            input_shape,
            num_classes,
            batchnorm=False,
            variational='full'):
  """Wrapper for build_resnet_v1.

  Args:
    n_examples (int): number of training points.
    input_shape (list): input shape.
    num_classes (int): number of classes (CIFAR10 has 10).
    batchnorm (bool): use of batchnorm layers.
    variational (str): 'none', 'hybrid', 'full'. whether to use variational
      inference for zero, some, or all layers.

  Returns:
      model (Model): Keras model instance whose output is a
        tfp.distributions.Categorical distribution.
  """
  inputs = tf.python.keras.layers.Input(shape=input_shape)
  x = build_resnet_v1(
      inputs,
      depth=20,
      variational=variational,
      batchnorm=batchnorm,
      n_examples=n_examples)

  p_fn, q_fn = mean_field_fn(empirical_bayes=True)

  def normalized_kl_fn(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.to_float(n_examples)

  logits = tfp.layers.DenseLocalReparameterization(
      num_classes,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(x)
  outputs = tf.python.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(logits)
  return tf.python.keras.models.Model(inputs=inputs, outputs=outputs)
