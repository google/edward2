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

"""EfficientNet adopted from official estimator version for tf2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import logging
import efficientnet_model  # local file import
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


# TODO(ywenxu): Check out `tf.keras.layers.experimental.SyncBatchNormalization.
# SyncBatchNorm on TPU. Orginal authored by hyhieu.
class SyncBatchNorm(tf.keras.layers.Layer):
  """BatchNorm that averages over ALL replicas. Only works for `NHWC` inputs."""

  def __init__(self, axis=3, momentum=0.99, epsilon=0.001,
               trainable=True, name='batch_norm', **kwargs):
    super(SyncBatchNorm, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon

  def build(self, input_shape):
    """Build function."""
    dim = input_shape[-1]
    shape = [dim]

    self.gamma = self.add_weight(
        name='gamma',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        trainable=True)

    self.beta = self.add_weight(
        name='beta',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        trainable=True)

    self.moving_mean = self.add_weight(
        name='moving_mean',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

    self.moving_variance = self.add_weight(
        name='moving_variance',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

  def _get_mean_and_variance(self, x):
    """Cross-replica mean and variance."""
    replica_context = tf.distribute.get_replica_context()
    num_replicas_in_sync = replica_context.num_replicas_in_sync
    if num_replicas_in_sync <= 8:
      group_assignment = None
      num_replicas_per_group = tf.cast(num_replicas_in_sync, tf.float32)
    else:
      num_replicas_per_group = max(8, num_replicas_in_sync // 8)
      group_assignment = np.arange(num_replicas_in_sync, dtype=np.int32)
      group_assignment = group_assignment.reshape([-1, num_replicas_per_group])
      group_assignment = group_assignment.tolist()
      num_replicas_per_group = tf.cast(num_replicas_per_group, tf.float32)

    mean = tf.reduce_mean(x, axis=[0, 1, 2])
    mean = tf.cast(mean, tf.float32)
    mean = tf1.tpu.cross_replica_sum(mean, group_assignment)
    mean = mean / num_replicas_per_group

    # Var[x] = E[x^2] - E[x]^2
    mean_sq = tf.reduce_mean(tf.square(x), axis=[0, 1, 2])
    mean_sq = tf.cast(mean_sq, tf.float32)
    mean_sq = tf1.tpu.cross_replica_sum(mean_sq, group_assignment)
    mean_sq = mean_sq / num_replicas_per_group
    variance = mean_sq - tf.square(mean)

    def _assign(moving, normal):
      decay = tf.cast(1. - self.momentum, tf.float32)
      diff = tf.cast(moving, tf.float32) - tf.cast(normal, tf.float32)
      return moving.assign_sub(decay * diff)

    self.add_update(_assign(self.moving_mean, mean))
    self.add_update(_assign(self.moving_variance, variance))

    # TODO(ywenxu): Assuming bfloat16. Fix for non bfloat16 case.
    mean = tf.cast(mean, tf.bfloat16)
    variance = tf.cast(variance, tf.bfloat16)

    return mean, variance

  def call(self, inputs, training):
    """Call function."""
    if training:
      mean, variance = self._get_mean_and_variance(inputs)
    else:
      mean, variance = self.moving_mean, self.moving_variance
    x = tf.nn.batch_normalization(
        inputs,
        mean=mean,
        variance=variance,
        offset=self.beta,
        scale=self.gamma,
        variance_epsilon=tf.cast(self.epsilon, variance.dtype),
    )
    return x


def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
      'efficientnet-b8': (2.2, 3.6, 672, 0.5),
      'efficientnet-l2': (4.3, 5.3, 800, 0.5),
  }
  return params_dict[model_name]


def build_model(width_coefficient,
                depth_coefficient,
                dropout_rate):
  """Creates model with default arguments.

  Args:
    width_coefficient: Coefficient to scale width.
    depth_coefficient: Coefficient to scale depth.
    dropout_rate: Dropout rate.

  Returns:
    tf.keras.Model.
  """
  global_params = efficientnet_model.GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      survival_prob=0.8,
      data_format='channels_last',
      num_classes=1000,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None,
      relu_fn=tf.nn.swish,
      batch_norm=SyncBatchNorm,  # TPU-specific requirement.
      use_se=True,
      clip_projection_output=False)
  logging.info('global_params= %s', global_params)
  BlockArgs = functools.partial(efficientnet_model.BlockArgs,  # pylint: disable=invalid-name
                                se_ratio=0.25)
  blocks_args = [
      BlockArgs(kernel_size=3,
                num_repeat=1,
                input_filters=32,
                output_filters=16,
                expand_ratio=1,
                strides=[1, 1]),
      BlockArgs(kernel_size=3,
                num_repeat=2,
                input_filters=16,
                output_filters=24,
                expand_ratio=6,
                strides=[2, 2]),
      BlockArgs(kernel_size=5,
                num_repeat=2,
                input_filters=24,
                output_filters=40,
                expand_ratio=6,
                strides=[2, 2]),
      BlockArgs(kernel_size=3,
                num_repeat=3,
                input_filters=40,
                output_filters=80,
                expand_ratio=6,
                strides=[2, 2]),
      BlockArgs(kernel_size=5,
                num_repeat=3,
                input_filters=80,
                output_filters=112,
                expand_ratio=6,
                strides=[1, 1]),
      BlockArgs(kernel_size=5,
                num_repeat=4,
                input_filters=112,
                output_filters=192,
                expand_ratio=6,
                strides=[2, 2]),
      BlockArgs(kernel_size=3,
                num_repeat=1,
                input_filters=192,
                output_filters=320,
                expand_ratio=6,
                strides=[1, 1]),
  ]
  return efficientnet_model.Model(blocks_args, global_params)
