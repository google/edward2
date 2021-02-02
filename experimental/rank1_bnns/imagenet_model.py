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
"""ResNet-50 with rank-1 distributions."""
import functools
import string
import edward2 as ed
from experimental.rank1_bnns import utils  # local file import
import tensorflow as tf

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def bottleneck_block(inputs,
                     filters,
                     stage,
                     block,
                     strides,
                     alpha_initializer,
                     gamma_initializer,
                     alpha_regularizer,
                     gamma_regularizer,
                     use_additive_perturbation,
                     ensemble_size,
                     random_sign_init,
                     dropout_rate,
                     prior_stddev,
                     use_tpu):
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
    prior_stddev: Standard deviation of the prior.
    use_tpu: whether the model runs on TPU.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = ed.layers.Conv2DRank1(
      filters1,
      kernel_size=1,
      use_bias=False,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2a',
      ensemble_size=ensemble_size)(inputs)
  x = ed.layers.ensemble_batchnorm(
      x,
      ensemble_size=ensemble_size,
      use_tpu=use_tpu,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base+'2a')
  x = tf.python.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DRank1(
      filters2,
      kernel_size=3,
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
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2b',
      ensemble_size=ensemble_size)(x)
  x = ed.layers.ensemble_batchnorm(
      x,
      ensemble_size=ensemble_size,
      use_tpu=use_tpu,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base+'2b')
  x = tf.python.keras.layers.Activation('relu')(x)

  x = ed.layers.Conv2DRank1(
      filters3,
      kernel_size=1,
      use_bias=False,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name=conv_name_base + '2c',
      ensemble_size=ensemble_size)(x)
  x = ed.layers.ensemble_batchnorm(
      x,
      ensemble_size=ensemble_size,
      use_tpu=use_tpu,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name=bn_name_base+'2c')

  shortcut = inputs
  if not x.shape.is_compatible_with(shortcut.shape):
    shortcut = ed.layers.Conv2DRank1(
        filters3,
        kernel_size=1,
        strides=strides,
        use_bias=False,
        alpha_initializer=utils.make_initializer(alpha_initializer,
                                                 random_sign_init,
                                                 dropout_rate),
        gamma_initializer=utils.make_initializer(gamma_initializer,
                                                 random_sign_init,
                                                 dropout_rate),
        kernel_initializer='he_normal',
        alpha_regularizer=utils.make_regularizer(
            alpha_regularizer, 1., prior_stddev),
        gamma_regularizer=utils.make_regularizer(
            gamma_regularizer, 1., prior_stddev),
        use_additive_perturbation=use_additive_perturbation,
        name=conv_name_base + '1',
        ensemble_size=ensemble_size)(inputs)
    shortcut = ed.layers.ensemble_batchnorm(
        shortcut,
        ensemble_size=ensemble_size,
        use_tpu=use_tpu,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=bn_name_base+'1')

  x = tf.python.keras.layers.add([x, shortcut])
  x = tf.python.keras.layers.Activation('relu')(x)
  return x


def group(inputs,
          filters,
          num_blocks,
          stage,
          strides,
          alpha_initializer,
          gamma_initializer,
          alpha_regularizer,
          gamma_regularizer,
          use_additive_perturbation,
          ensemble_size,
          random_sign_init,
          dropout_rate,
          prior_stddev,
          use_tpu):
  """Group of residual blocks."""
  bottleneck_block_ = functools.partial(
      bottleneck_block,
      filters=filters,
      stage=stage,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate,
      prior_stddev=prior_stddev,
      use_tpu=use_tpu)
  blocks = string.ascii_lowercase
  x = bottleneck_block_(inputs, block=blocks[0], strides=strides)
  for i in range(num_blocks - 1):
    x = bottleneck_block_(x, block=blocks[i + 1], strides=1)
  return x


def rank1_resnet50(input_shape,
                   num_classes,
                   alpha_initializer,
                   gamma_initializer,
                   alpha_regularizer,
                   gamma_regularizer,
                   use_additive_perturbation,
                   ensemble_size,
                   random_sign_init,
                   dropout_rate,
                   prior_stddev,
                   use_tpu):
  """Builds ResNet50 with rank 1 priors.

  Using strided conv, pooling, four groups of residual blocks, and pooling, the
  network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
  14x14 -> 7x7 (Table 1 of He et al. (2015)).

  Args:
    input_shape: Shape tuple of input excluding batch dimension.
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
    prior_stddev: Standard deviation of the prior.
    use_tpu: whether the model runs on TPU.

  Returns:
    tf.python.keras.Model.
  """
  group_ = functools.partial(
      group,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      alpha_regularizer=alpha_regularizer,
      gamma_regularizer=gamma_regularizer,
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      random_sign_init=random_sign_init,
      dropout_rate=dropout_rate,
      prior_stddev=prior_stddev,
      use_tpu=use_tpu)
  inputs = tf.python.keras.layers.Input(shape=input_shape)
  x = tf.python.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(inputs)
  x = ed.layers.Conv2DRank1(
      64,
      kernel_size=7,
      strides=2,
      padding='valid',
      use_bias=False,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer='he_normal',
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      name='conv1',
      ensemble_size=ensemble_size)(x)
  x = ed.layers.ensemble_batchnorm(
      x,
      ensemble_size=ensemble_size,
      use_tpu=use_tpu,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name='bn_conv1')
  x = tf.python.keras.layers.Activation('relu')(x)
  x = tf.python.keras.layers.MaxPooling2D(3, strides=(2, 2), padding='same')(x)
  x = group_(x, [64, 64, 256], stage=2, num_blocks=3, strides=1)
  x = group_(x, [128, 128, 512], stage=3, num_blocks=4, strides=2)
  x = group_(x, [256, 256, 1024], stage=4, num_blocks=6, strides=2)
  x = group_(x, [512, 512, 2048], stage=5, num_blocks=3, strides=2)
  x = tf.python.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.DenseRank1(
      num_classes,
      alpha_initializer=utils.make_initializer(alpha_initializer,
                                               random_sign_init,
                                               dropout_rate),
      gamma_initializer=utils.make_initializer(gamma_initializer,
                                               random_sign_init,
                                               dropout_rate),
      kernel_initializer=tf.python.keras.initializers.RandomNormal(stddev=0.01),
      alpha_regularizer=utils.make_regularizer(
          alpha_regularizer, 1., prior_stddev),
      gamma_regularizer=utils.make_regularizer(
          gamma_regularizer, 1., prior_stddev),
      use_additive_perturbation=use_additive_perturbation,
      ensemble_size=ensemble_size,
      activation=None,
      name='fc1000')(x)
  return tf.python.keras.Model(inputs=inputs, outputs=x, name='resnet50')
