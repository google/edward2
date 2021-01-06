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

# Lint as: python3
"""Wide ResNet architecture with multiple input and outputs."""
import functools
from experimental.mimo import layers  # local file import
import tensorflow as tf

BATCHNORM_L2 = 3e-4

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9,
    beta_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2),
    gamma_regularizer=tf.keras.regularizers.l2(BATCHNORM_L2))
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')

l1_l2 = tf.keras.regularizers.l1_l2


def basic_block(inputs, filters, strides, l2=0., l1=0.):
  """Basic residual block of two 3x3 convs."""

  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=strides,
             kernel_regularizer=l1_l2(l1=l1, l2=l2))(y)
  y = BatchNormalization()(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=1,
             kernel_regularizer=l1_l2(l1=l1, l2=l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters, kernel_size=1, strides=strides,
               kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, l2=0., l1=0., **kwargs):
  """Group of residual blocks."""
  x = basic_block(
      inputs, filters=filters, strides=strides, l2=l2, l1=l1, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, l2=l2, l1=l1, **kwargs)
  return x


def wide_resnet(input_shape, depth, width_multiplier, num_classes,
                ensemble_size, l2=0., l1=0.):
  """Builds Wide ResNet with Sparse BatchEnsemble.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor. The input shape must be (ensemble_size, width,
      height, channels).
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    ensemble_size: Number of ensemble members.
    l2: L2 regularization value.
    l1: L1 regularization value.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  input_shape = list(input_shape)
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.Permute([2, 3, 4, 1])(inputs)
  if ensemble_size != input_shape[0]:
    raise ValueError('the first dimension of input_shape must be ensemble_size')
  x = tf.keras.layers.Reshape(input_shape[1:-1] +
                              [input_shape[-1] * ensemble_size])(x)
  # since the first conv layer and the last dense layer have ensemble_size more
  # weights, we multiply the regularization coefficients by that amount
  scaled_l2 = l2 * ensemble_size
  scaled_l1 = l1 * ensemble_size
  kernel_regularizer = l1_l2(l1=scaled_l1, l2=scaled_l2)
  x = Conv2D(16, strides=1, kernel_regularizer=kernel_regularizer)(x)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(
        x,
        filters=filters * width_multiplier,
        strides=strides,
        num_blocks=num_blocks,
        l2=l2,
        l1=l1)

  x = BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = layers.DenseMultihead(
      num_classes,
      kernel_initializer='he_normal',
      activation=None,
      ensemble_size=ensemble_size,
      kernel_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2),
      bias_regularizer=l1_l2(l1=scaled_l1, l2=scaled_l2))(x)
  return tf.keras.Model(inputs=inputs, outputs=x)
