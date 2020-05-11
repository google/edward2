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

"""ResNet50 model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import string

import tensorflow.compat.v2 as tf


# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


Conv2DBase = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


# pylint: disable=invalid-name
def bottleneck_block(inputs, filters, stage, block, strides, Conv2D):
  """Residual block with 1x1 -> 3x3 -> 1x1 convs in main path.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    inputs: tf.Tensor.
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    Conv2D: The Conv2D function for constructing convolutional layers.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = Conv2D(
      filters1,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a')(inputs)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2a')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2D(
      filters2,
      kernel_size=3,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2b')(x)
  x = tf.keras.layers.Activation('relu')(x)

  x = Conv2D(
      filters3,
      kernel_size=1,
      use_bias=False,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base + '2c')(x)

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = Conv2D(
        filters3,
        kernel_size=1,
        use_bias=False,
        strides=strides,
        kernel_initializer='he_normal',
        name=conv_name_base + '1')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization(
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base + '1')(shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, num_blocks, stage, strides, Conv2D):
  blocks = string.ascii_lowercase
  x = bottleneck_block(
      inputs, filters, stage, block=blocks[0], strides=strides, Conv2D=Conv2D)
  for i in range(num_blocks - 1):
    x = bottleneck_block(
        x, filters, stage, block=blocks[i + 1], strides=1, Conv2D=Conv2D)
  return x


def resnet50(input_shape, num_classes, batch_size, use_ngp_layer, use_spec_norm,
             global_step, sn_iteration, sn_norm_bound, gp_input_dim,
             **ngp_kwargs):
  """Builds ResNet50.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
    num_classes: Number of output classes.
    batch_size:  Number of samples in a minibatch.
    use_ngp_layer: (bool) Whether to use Normalized GP as output layer.
    use_spec_norm: (bool) Whether to use spectral normalization on conv layers.
    global_step: (tf.Variable) The total number of accumulated steps.
    sn_iteration: (int) Number of power iteration for spec normalization.
    sn_norm_bound: (float) Upper bound on singular value in spec normalization.
    gp_input_dim: (int) Input dimension to NormalizedGaussianProcess.
    **ngp_kwargs: Keyword arguments to NormalizedGaussianProcess layer.

  Returns:
    tf.keras.Model.
  """
  # define convolutional layer.
  def Conv2DNormed(*conv_args, **conv_kwargs):
    conv_layer = Conv2DBase(*conv_args, **conv_kwargs)
    return sn.SpectralNormalizationConv2D(
        conv_layer, iteration=sn_iteration, norm_multiplier=sn_norm_bound)

  Conv2D = Conv2DNormed if use_spec_norm else Conv2DBase

  # Define network.
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = Conv2D(
      64,
      kernel_size=7,
      strides=2,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal',
      name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
  x = group(x, [64, 64, 256], stage=2, num_blocks=3, strides=1, Conv2D=Conv2D)
  x = group(x, [128, 128, 512], stage=3, num_blocks=4, strides=2, Conv2D=Conv2D)
  x = group(
      x, [256, 256, 1024], stage=4, num_blocks=6, strides=2, Conv2D=Conv2D)
  x = group(
      x, [512, 512, 2048], stage=5, num_blocks=3, strides=2, Conv2D=Conv2D)
  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)

  if use_ngp_layer:
    # add random projection layer to reduce dimension
    x = tf.keras.layers.Dense(
        gp_input_dim, kernel_initializer='random_normal', use_bias=False)(
            x)

    x, x_stddev = ngp.NormalizedGaussianProcess(num_classes,
                                                **ngp_kwargs)(x, global_step)
  else:
    x = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        name='fc1000')(x)
    x_stddev = tf.ones(shape=(batch_size,))

  return tf.keras.Model(inputs=inputs, outputs=[x, x_stddev], name='resnet50')
