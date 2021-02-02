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
"""Wide ResNet architecture with rank-1 distributions."""
import functools
import edward2 as ed
from experimental.rank1_bnns import utils  # local file import
import tensorflow as tf

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.python.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2DRank1 = functools.partial(  # pylint: disable=invalid-name
    ed.layers.Conv2DRank1,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def basic_block(inputs,
                filters,
                strides,
                alpha_initializer,
                gamma_initializer,
                alpha_regularizer,
                gamma_regularizer,
                use_additive_perturbation,
                ensemble_size,
                random_sign_init,
                dropout_rate,
                prior_mean,
                prior_stddev):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    alpha_initializer: The initializer for the alpha parameters.
    gamma_initializer: The initializer for the gamma parameters.
    alpha_regularizer: The regularizer for the alpha parameters.
    gamma_regularizer: The regularizer for the gamma parameters.
    use_additive_perturbation: Whether or not to use additive perturbations
      instead of multiplicative perturbations.
    ensemble_size: Number of ensemble members.
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.
    dropout_rate: Dropout rate.
    prior_mean: Mean of the prior.
    prior_stddev: Standard deviation of the prior.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization()(y)
  y = tf.python.keras.layers.Activation('relu')(y)
  y = Conv2DRank1(
      filters,
      strides=strides,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(y)
  y = BatchNormalization()(y)
  y = tf.python.keras.layers.Activation('relu')(y)
  y = Conv2DRank1(
      filters,
      strides=1,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2DRank1(
        filters,
        kernel_size=1,
        strides=strides,
        alpha_initializer=utils.make_initializer(alpha_initializer,
                                                 random_sign_init,
                                                 dropout_rate),
        gamma_initializer=utils.make_initializer(gamma_initializer,
                                                 random_sign_init,
                                                 dropout_rate),
        alpha_regularizer=utils.make_regularizer(
            alpha_regularizer, prior_mean, prior_stddev),
        gamma_regularizer=utils.make_regularizer(
            gamma_regularizer, prior_mean, prior_stddev),
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)(x)
  x = tf.python.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def wide_resnet(input_shape,
                depth,
                width_multiplier,
                num_classes,
                alpha_initializer,
                gamma_initializer,
                alpha_regularizer,
                gamma_regularizer,
                use_additive_perturbation,
                ensemble_size,
                random_sign_init,
                dropout_rate,
                prior_mean,
                prior_stddev):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    alpha_initializer: The initializer for the alpha parameters.
    gamma_initializer: The initializer for the gamma parameters.
    alpha_regularizer: The regularizer for the alpha parameters.
    gamma_regularizer: The regularizer for the gamma parameters.
    use_additive_perturbation: Whether or not to use additive perturbations
      instead of multiplicative perturbations.
    ensemble_size: Number of ensemble members.
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.
    dropout_rate: Dropout rate.
    prior_mean: Mean of the prior.
    prior_stddev: Standard deviation of the prior.

  Returns:
    tf.python.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.python.keras.layers.Input(shape=input_shape)
  x = Conv2DRank1(
      16,
      strides=1,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(inputs)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(x,
              filters=filters * width_multiplier,
              strides=strides,
              num_blocks=num_blocks,
              alpha_initializer=alpha_initializer,
              gamma_initializer=gamma_initializer,
              alpha_regularizer=alpha_regularizer,
              gamma_regularizer=gamma_regularizer,
              use_additive_perturbation=use_additive_perturbation,
              ensemble_size=ensemble_size,
              random_sign_init=random_sign_init,
              dropout_rate=dropout_rate,
              prior_mean=prior_mean,
              prior_stddev=prior_stddev)

  x = BatchNormalization()(x)
  x = tf.python.keras.layers.Activation('relu')(x)
  x = tf.python.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.python.keras.layers.Flatten()(x)
  x = ed.layers.DenseRank1(
      num_classes,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      activation=None,
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, prior_mean, prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, prior_mean, prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(x)
  return tf.python.keras.Model(inputs=inputs, outputs=x)
