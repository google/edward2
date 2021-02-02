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
"""ResNet-32x4 with rank-1 distributions on CIFAR-10 and CIFAR-100.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
import functools
import edward2 as ed
from experimental.rank1_bnns import utils  # local file import
import tensorflow as tf

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def rank1_resnet_layer(inputs,
                       filters,
                       kernel_size,
                       strides,
                       activation,
                       alpha_initializer,
                       gamma_initializer,
                       alpha_regularizer,
                       gamma_regularizer,
                       use_additive_perturbation,
                       ensemble_size,
                       random_sign_init,
                       dropout_rate):
  """Bayesian rank-1 2D Convolution-Batch Norm-Activation stack builder.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    kernel_size: Kernel dimensions for Conv2D.
    strides: Stride dimensinons for Conv2D.
    activation: tf.python.keras.activations.Activation.
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

  Returns:
    tf.Tensor.
  """
  x = inputs
  x = ed.layers.Conv2DRank1(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(x)
  x = tf.python.keras.layers.BatchNormalization(epsilon=BATCH_NORM_EPSILON,
                                         momentum=BATCH_NORM_DECAY)(x)
  if activation is not None:
    x = tf.python.keras.layers.Activation(activation)(x)
  return x


def rank1_resnet_v1(input_shape,
                    depth,
                    num_classes,
                    width_multiplier,
                    alpha_initializer,
                    gamma_initializer,
                    alpha_regularizer,
                    gamma_regularizer,
                    use_additive_perturbation,
                    ensemble_size,
                    random_sign_init,
                    dropout_rate):
  """Builds Bayesian rank-1 prior ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    width_multiplier: Integer to multiply the number of typical filters by.
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

  Returns:
    tf.python.keras.Model.
  """
  if (depth - 2) % 6 != 0:
    raise ValueError('depth should be 6n+2 (e.g., 20, 32, 44).')
  filters = 16 * width_multiplier
  num_res_blocks = int((depth - 2) / 6)

  resnet_layer = functools.partial(
      rank1_resnet_layer,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate)
  inputs = tf.python.keras.layers.Input(shape=input_shape)
  x = resnet_layer(inputs,
                   filters=filters,
                   kernel_size=3,
                   strides=1,
                   activation='relu')
  for stack in range(3):
    for res_block in range(num_res_blocks):
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(
          x,
          filters=filters,
          kernel_size=3,
          strides=strides,
          activation='relu')
      y = resnet_layer(
          y,
          filters=filters,
          kernel_size=3,
          strides=1,
          activation=None)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match
        # changed dims
        x = resnet_layer(
            x,
            filters=filters,
            kernel_size=1,
            strides=strides,
            activation=None)
      x = tf.python.keras.layers.add([x, y])
      x = tf.python.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.python.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.python.keras.layers.Flatten()(x)
  x = ed.layers.DenseRank1(
      num_classes,
      activation=None,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size)(x)
  model = tf.python.keras.Model(inputs=inputs, outputs=x)
  return model
