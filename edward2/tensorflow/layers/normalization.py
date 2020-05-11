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

"""Normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow import random_variable
from edward2.tensorflow import transformed_random_variable
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


class ActNorm(tf.keras.layers.Layer):
  """Actnorm, an affine reversible layer (Prafulla and Kingma, 2018).

  Weights use data-dependent initialization in which outputs have zero mean
  and unit variance per channel (last dimension). The mean/variance statistics
  are computed from the first batch of inputs.
  """

  def __init__(self, epsilon=tf.keras.backend.epsilon(), **kwargs):
    super(ActNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `ActNorm` '
                       'should be defined. Found `None`.')
    bias = self.add_weight('bias', [last_dim], dtype=self.dtype)
    log_scale = self.add_weight('log_scale', [last_dim], dtype=self.dtype)
    # Set data-dependent initializers.
    bias = bias.assign(self.bias_initial_value)
    with tf.control_dependencies([bias]):
      self.bias = bias
    log_scale = log_scale.assign(self.log_scale_initial_value)
    with tf.control_dependencies([log_scale]):
      self.log_scale = log_scale
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not self.built:
      mean, variance = tf.nn.moments(
          inputs, axes=list(range(inputs.shape.ndims - 1)))
      self.bias_initial_value = -mean
      # TODO(trandustin): Optionally, actnorm multiplies log_scale by a fixed
      # log_scale factor (e.g., 3.) and initializes by
      # initial_value / log_scale_factor.
      self.log_scale_initial_value = tf.math.log(
          1. / (tf.sqrt(variance) + self.epsilon))

    if not isinstance(inputs, random_variable.RandomVariable):
      return super(ActNorm, self).__call__(inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs):
    return (inputs + self.bias) * tf.exp(self.log_scale)

  def reverse(self, inputs):
    return inputs * tf.exp(-self.log_scale) - self.bias

  def log_det_jacobian(self, inputs):
    """Returns log det | dx / dy | = num_events * sum log | scale |."""
    # Number of events is number of all elements excluding the batch and
    # channel dimensions.
    num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
    log_det_jacobian = num_events * tf.reduce_sum(self.log_scale)
    return log_det_jacobian


def ensemble_batchnorm(x, ensemble_size=1, use_tpu=True, **kwargs):
  """A modified batch norm layer for Batch Ensemble model.

  Args:
    x: input tensor.
    ensemble_size: number of ensemble members.
    use_tpu: whether the model is running on TPU.
    **kwargs: Keyword arguments to batch normalization layers.

  Returns:
    Output tensor for the block.
  """
  # In BatchEnsemble inference stage, the input to the model is tiled which
  # leads to dynamic shape because of the tf.split function. Such operation
  # is not supported in tf2.0 on TPU. For current workaround, we use single
  # BatchNormalization layer for all ensemble member. This is not correct in
  # math but works in practice.
  if ensemble_size == 1 or use_tpu:
    return tf.keras.layers.BatchNormalization(**kwargs)(x)
  name = kwargs.get('name')
  split_inputs = tf.split(x, ensemble_size, axis=0)
  for i in range(ensemble_size):
    if name is not None:
      kwargs['name'] = name + '_{}'.format(i)
    split_inputs[i] = tf.keras.layers.BatchNormalization(**kwargs)(
        split_inputs[i])
  return tf.concat(split_inputs, axis=0)


class EnsembleSyncBatchNorm(tf.keras.layers.Layer):
  """BatchNorm that averages over ALL replicas. Only works for `NHWC` inputs."""

  def __init__(self, axis=3, ensemble_size=1, momentum=0.99, epsilon=0.001,
               trainable=True, name=None, **kwargs):
    super(EnsembleSyncBatchNorm, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.ensemble_size = ensemble_size

  def build(self, input_shape):
    """Build function."""
    dim = input_shape[-1]
    if self.ensemble_size > 1:
      shape = [self.ensemble_size, dim]
    else:
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

    if replica_context is not None:
      num_replicas_in_sync = replica_context.num_replicas_in_sync
      if num_replicas_in_sync <= 8:
        group_assignment = None
        num_replicas_per_group = tf.cast(num_replicas_in_sync, tf.float32)
      else:
        num_replicas_per_group = max(8, num_replicas_in_sync // 8)
        group_assignment = np.arange(num_replicas_in_sync, dtype=np.int32)
        group_assignment = group_assignment.reshape(
            [-1, num_replicas_per_group])
        group_assignment = group_assignment.tolist()
        num_replicas_per_group = tf.cast(num_replicas_per_group, tf.float32)

    # This only supports NHWC format.
    if self.ensemble_size > 1:
      height = tf.shape(x)[1]
      width = tf.shape(x)[2]
      input_channels = tf.shape(x)[3]
      x = tf.reshape(x, [self.ensemble_size, -1, height, width, input_channels])
      mean = tf.reduce_mean(x, axis=[1, 2, 3])  # [ensemble_size, channels]
      mean = tf.cast(mean, tf.float32)

      # Var[x] = E[x^2] - E[x]^2
      mean_sq = tf.reduce_mean(tf.square(x), axis=[1, 2, 3])
      mean_sq = tf.cast(mean_sq, tf.float32)
      if replica_context is not None:
        mean = tf1.tpu.cross_replica_sum(mean, group_assignment)
        mean = mean / num_replicas_per_group
        mean_sq = tf1.tpu.cross_replica_sum(mean_sq, group_assignment)
        mean_sq = mean_sq / num_replicas_per_group
      variance = mean_sq - tf.square(mean)
    else:
      mean = tf.reduce_mean(x, axis=[0, 1, 2])
      mean = tf.cast(mean, tf.float32)

      mean_sq = tf.reduce_mean(tf.square(x), axis=[0, 1, 2])
      mean_sq = tf.cast(mean_sq, tf.float32)
      if replica_context is not None:
        mean = tf1.tpu.cross_replica_sum(mean, group_assignment)
        mean = mean / num_replicas_per_group
        mean_sq = tf1.tpu.cross_replica_sum(mean_sq, group_assignment)
        mean_sq = mean_sq / num_replicas_per_group
      variance = mean_sq - tf.square(mean)

    def _assign(moving, normal):
      decay = tf.cast(1. - self.momentum, tf.float32)
      diff = tf.cast(moving, tf.float32) - tf.cast(normal, tf.float32)
      return moving.assign_sub(decay * diff)

    self.add_update(_assign(self.moving_mean, mean))
    self.add_update(_assign(self.moving_variance, variance))

    mean = tf.cast(mean, x.dtype)
    variance = tf.cast(variance, x.dtype)

    return mean, variance

  def call(self, inputs, training):
    """Call function."""
    if training:
      mean, variance = self._get_mean_and_variance(inputs)
    else:
      mean, variance = self.moving_mean, self.moving_variance
    if self.ensemble_size > 1:
      batch_size = tf.shape(inputs)[0]
      input_dim = tf.shape(mean)[-1]
      examples_per_model = batch_size // self.ensemble_size
      mean = tf.reshape(tf.tile(mean, [1, examples_per_model]),
                        [batch_size, input_dim])
      variance_epsilon = tf.cast(self.epsilon, variance.dtype)
      inv = tf.math.rsqrt(variance + variance_epsilon)
      if self.gamma is not None:
        inv *= self.gamma
      inv = tf.reshape(tf.tile(inv, [1, examples_per_model]),
                       [batch_size, input_dim])
      # Assuming channel last.
      inv = tf.expand_dims(inv, axis=1)
      inv = tf.expand_dims(inv, axis=1)
      mean = tf.expand_dims(mean, axis=1)
      mean = tf.expand_dims(mean, axis=1)
      if self.beta is not None:
        beta = tf.reshape(tf.tile(self.beta, [1, examples_per_model]),
                          [batch_size, input_dim])
        beta = tf.expand_dims(beta, axis=1)
        beta = tf.expand_dims(beta, axis=1)
      x = inputs * tf.cast(inv, inputs.dtype) + tf.cast(
          beta - mean * inv if self.beta is not None else (
              -mean * inv), inputs.dtype)
    else:
      x = tf.nn.batch_normalization(
          inputs,
          mean=mean,
          variance=variance,
          offset=self.beta,
          scale=self.gamma,
          variance_epsilon=tf.cast(self.epsilon, variance.dtype),
      )
    return x
