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

"""Normalization layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow import random_variable
from edward2.tensorflow import transformed_random_variable
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
    if isinstance(last_dim, tf1.Dimension):
      last_dim = last_dim.value
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
    del inputs  # unused
    # Number of events is number of all elements excluding the batch and
    # channel dimensions.
    num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
    log_det_jacobian = num_events * tf.reduce_sum(self.log_scale)
    return log_det_jacobian


def ensemble_batchnorm(x, num_models=1, use_tpu=True, **kwargs):
  """A modified batch norm layer for Batch Ensemble model.

  Args:
    x: input tensor.
    num_models: number of ensemble members.
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
  if num_models == 1 or use_tpu:
    return tf.keras.layers.BatchNormalization(**kwargs)(x)
  name = kwargs.get('name')
  split_inputs = tf.split(x, num_models, axis=0)
  for i in range(num_models):
    if name is not None:
      kwargs['name'] = name + '_{}'.format(i)
    split_inputs[i] = tf.keras.layers.BatchNormalization(**kwargs)(
        split_inputs[i])
  return tf.concat(split_inputs, axis=0)
