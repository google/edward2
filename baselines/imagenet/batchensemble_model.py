# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

"""Batch Ensemble ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().
and //third_party/cloud_tpu/models/resnet50_keras.

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import tensorflow.compat.v2 as tf

# Use batch normalization defaults from Pytorch.
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def ensemble_identity_block(input_tensor,
                            kernel_size,
                            filters,
                            stage,
                            block,
                            num_models,
                            random_sign_init=1.0,
                            use_tpu=True):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: tf.Tensor.
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    num_models: the ensemble size, when it is one, it goes back to the
        single model case.
    random_sign_init (float): if random_sign_init > 0, fast weight is
        intialized as +/- 1 with probability random_sign_init. If
        random_sign_init < 0, fast weight is generated with Gaussian mean 1,
        std -random_sign_init.
    use_tpu: whether the model runs on TPU.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  if random_sign_init > 0:
    alpha_initializer = ed.initializers.RandomSign(random_sign_init)
    gamma_initializer = ed.initializers.RandomSign(random_sign_init)
  else:
    alpha_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)
    gamma_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)

  x = ed.layers.BatchEnsembleConv2D(
      filters1, (1, 1),
      use_bias=False,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a',
      num_models=num_models)(input_tensor)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2a',
      num_models=num_models,
      use_tpu=use_tpu)

  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.BatchEnsembleConv2D(
      filters2, kernel_size,
      use_bias=False,
      padding='same',
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b',
      num_models=num_models)(x)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2b',
      num_models=num_models,
      use_tpu=use_tpu)

  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.BatchEnsembleConv2D(
      filters3, (1, 1),
      use_bias=False,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c',
      num_models=num_models)(x)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2c',
      num_models=num_models,
      use_tpu=use_tpu)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def ensemble_conv_block(input_tensor,
                        kernel_size,
                        filters,
                        stage,
                        block,
                        strides=(2, 2),
                        num_models=1,
                        random_sign_init=False,
                        use_tpu=True):
  """A block that has a conv layer at shortcut.

  Note that strides appear in the second conv (3x3) rather than the first (1x1).
  This is also known as "ResNet v1.5" as it differs from He et al. (2015)
  (http://torch.ch/blog/2016/02/04/resnets.html).

  Args:
    input_tensor: tf.Tensor.
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    num_models: the ensemble size, when it is one, it goes back to the
        single model case.
    random_sign_init: whether uses random sign initializer to initializer
        the fast weights.
    use_tpu: whether the model runs on TPU.

  Returns:
    tf.Tensor.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  if random_sign_init > 0:
    alpha_initializer = ed.initializers.RandomSign(random_sign_init)
    gamma_initializer = ed.initializers.RandomSign(random_sign_init)
  else:
    alpha_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)
    gamma_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)

  x = ed.layers.BatchEnsembleConv2D(
      filters1, (1, 1),
      use_bias=False,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2a',
      num_models=num_models)(input_tensor)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2a',
      num_models=num_models,
      use_tpu=use_tpu)
  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.BatchEnsembleConv2D(
      filters2, kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2b',
      num_models=num_models)(x)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2b',
      num_models=num_models,
      use_tpu=use_tpu)

  x = tf.keras.layers.Activation('relu')(x)

  x = ed.layers.BatchEnsembleConv2D(
      filters3, (1, 1),
      use_bias=False,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '2c',
      num_models=num_models)(x)

  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name=bn_name_base+'2c',
      num_models=num_models,
      use_tpu=use_tpu)

  shortcut = ed.layers.BatchEnsembleConv2D(
      filters3, (1, 1),
      use_bias=False,
      strides=strides,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name=conv_name_base + '1')(input_tensor)
  shortcut = ed.layers.ensemble_batchnorm(
      shortcut, axis=bn_axis,
      name=bn_name_base+'1',
      num_models=num_models,
      use_tpu=use_tpu)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def ensemble_resnet50(num_classes,
                      num_models=1,
                      random_sign_init=False,
                      use_tpu=True):
  """Instantiates the BatchEnsemble ResNet50 architecture.

  Args:
    num_classes: Number of output classes.
    num_models: Ensemble size.
    random_sign_init: float, probability of RandomSign initializer.
    use_tpu: whether the model runs on TPU.

  Returns:
    tf.keras.Model.
  """
  if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, 224, 224)
    bn_axis = 1
  else:
    input_shape = (224, 224, 3)
    bn_axis = 3

  if random_sign_init > 0:
    alpha_initializer = ed.initializers.RandomSign(random_sign_init)
    gamma_initializer = ed.initializers.RandomSign(random_sign_init)
  else:
    alpha_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)
    gamma_initializer = tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)

  img_input = tf.keras.layers.Input(shape=input_shape)
  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = ed.layers.BatchEnsembleConv2D(
      64, (7, 7),
      use_bias=False,
      strides=(2, 2),
      padding='valid',
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      kernel_initializer='he_normal',
      name='conv1',
      num_models=num_models)(x)
  x = ed.layers.ensemble_batchnorm(
      x, axis=bn_axis,
      name='bn_conv1',
      num_models=num_models,
      use_tpu=use_tpu)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  # TODO(ywenxu): use for loop to generate residual block.
  x = ensemble_conv_block(
      x, 3, [64, 64, 256],
      stage=2, block='a',
      strides=(1, 1),
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [64, 64, 256],
      stage=2, block='b',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [64, 64, 256],
      stage=2, block='c',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)

  x = ensemble_conv_block(
      x, 3, [128, 128, 512],
      stage=3, block='a',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [128, 128, 512],
      stage=3, block='b',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [128, 128, 512],
      stage=3, block='c',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [128, 128, 512],
      stage=3, block='d',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)

  x = ensemble_conv_block(
      x, 3, [256, 256, 1024],
      stage=4, block='a',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [256, 256, 1024],
      stage=4, block='b',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [256, 256, 1024],
      stage=4, block='c',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [256, 256, 1024],
      stage=4, block='d',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [256, 256, 1024],
      stage=4, block='e',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [256, 256, 1024],
      stage=4, block='f',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)

  x = ensemble_conv_block(
      x, 3, [512, 512, 2048],
      stage=5, block='a',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [512, 512, 2048],
      stage=5, block='b',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)
  x = ensemble_identity_block(
      x, 3, [512, 512, 2048],
      stage=5, block='c',
      num_models=num_models,
      use_tpu=use_tpu,
      random_sign_init=random_sign_init)

  x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  x = ed.layers.BatchEnsembleDense(
      num_classes,
      num_models=num_models,
      alpha_initializer=alpha_initializer,
      gamma_initializer=gamma_initializer,
      activation='softmax',
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      name='fc1000')(x)
  return tf.keras.models.Model(img_input, x, name='ensemble_resnet50')
