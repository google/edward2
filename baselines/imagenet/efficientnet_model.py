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

"""EfficientNet model adopted from official estimator version for tf2.

[1] Mingxing Tan, Quoc V. Le
  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
  ICML'19, https://arxiv.org/abs/1905.11946.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
import tensorflow.compat.v2 as tf

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient', 'depth_divisor',
    'min_depth', 'survival_prob', 'relu_fn', 'batch_norm', 'use_se',
    'clip_projection_output',
])
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'strides', 'se_ratio'
])


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.math.divide(inputs, survival_prob) * binary_tensor
  return output


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)


def round_filters(filters, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.width_coefficient
  divisor = global_params.depth_divisor
  min_depth = global_params.min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats, global_params, skip=False):
  """Round number of filters based on depth multiplier."""
  multiplier = global_params.depth_coefficient
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


class MBConvBlock(tf.keras.layers.Layer):
  """A class of MBConv: Mobile Inverted Residual Bottleneck."""

  def __init__(self, block_args, global_params):
    """Initializes a MBConv block.

    Args:
      block_args: BlockArgs, arguments to create a Block.
      global_params: GlobalParams, a set of global parameters.
    """
    super(MBConvBlock, self).__init__()
    self._block_args = block_args
    self._batch_norm_momentum = global_params.batch_norm_momentum
    self._batch_norm_epsilon = global_params.batch_norm_epsilon
    self._batch_norm = global_params.batch_norm
    self._data_format = global_params.data_format
    if self._data_format == 'channels_first':
      self._channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      self._channel_axis = -1
      self._spatial_dims = [1, 2]

    self._relu_fn = global_params.relu_fn
    self._has_se = (
        global_params.use_se and self._block_args.se_ratio is not None and
        0 < self._block_args.se_ratio <= 1)

    self._clip_projection_output = global_params.clip_projection_output

    self.conv_cls = tf.keras.layers.Conv2D
    self.depthwise_conv_cls = tf.keras.layers.DepthwiseConv2D

    # Builds the block accordings to arguments.
    self._build()

  def _build(self):
    """Builds block according to the arguments."""
    filters = self._block_args.input_filters * self._block_args.expand_ratio
    kernel_size = self._block_args.kernel_size

    # Expansion phase. Called if expansion phase is necessary.
    self._expand_conv = self.conv_cls(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False)
    self._bn0 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    # Depth-wise convolution phase.
    self._depthwise_conv = self.depthwise_conv_cls(
        kernel_size=[kernel_size, kernel_size],
        strides=self._block_args.strides,
        depthwise_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False)

    self._bn1 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

    if self._has_se:
      num_reduced_filters = max(
          1, int(self._block_args.input_filters * self._block_args.se_ratio))
      # Squeeze and Excitation layer.
      self._se_reduce = tf.keras.layers.Conv2D(
          num_reduced_filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=True)
      self._se_expand = tf.keras.layers.Conv2D(
          filters,
          kernel_size=[1, 1],
          strides=[1, 1],
          kernel_initializer=conv_kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=True)

    # Output phase.
    filters = self._block_args.output_filters
    self._project_conv = self.conv_cls(
        filters=filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False)
    self._bn2 = self._batch_norm(
        axis=self._channel_axis,
        momentum=self._batch_norm_momentum,
        epsilon=self._batch_norm_epsilon)

  def _call_se(self, input_tensor):
    """Call Squeeze and Excitation layer.

    Args:
      input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.

    Returns:
      A output tensor, which should have the same shape as input.
    """
    se_tensor = tf.reduce_mean(
        input_tensor, self._spatial_dims, keepdims=True)
    se_tensor = self._se_expand(self._relu_fn(self._se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * input_tensor

  def call(self, inputs, training=True, survival_prob=None):
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    x = inputs

    expand_conv_fn = self._expand_conv
    depthwise_conv_fn = self._depthwise_conv
    project_conv_fn = self._project_conv

    # Apply expansion and then apply depthwise conv.
    if self._block_args.expand_ratio != 1:
      x = self._relu_fn(self._bn0(expand_conv_fn(x), training=training))
    x = self._relu_fn(self._bn1(depthwise_conv_fn(x), training=training))

    if self._has_se:
      x = self._call_se(x)

    x = self._bn2(project_conv_fn(x), training=training)
    # Add identity so that quantization-aware training can insert quantization
    # ops correctly.
    x = tf.identity(x)
    if self._clip_projection_output:
      x = tf.clip_by_value(x, -6, 6)
    if all(
        s == 1 for s in self._block_args.strides
    ) and self._block_args.input_filters == self._block_args.output_filters:
      if survival_prob:
        x = drop_connect(x, training, survival_prob)
      x = tf.add(x, inputs)
    return x


class Model(tf.keras.Model):
  """A class implements tf.keras.Model for MNAS-like model.

    Reference: https://arxiv.org/abs/1807.11626
  """

  def __init__(self, blocks_args, global_params):
    """Initializes an `Model` instance.

    Args:
      blocks_args: A list of BlockArgs to construct block modules.
      global_params: GlobalParams, a set of global parameters.

    Raises:
      ValueError: when blocks_args is not specified as a list.
    """
    super(Model, self).__init__()
    if not isinstance(blocks_args, list):
      raise ValueError('blocks_args should be a list.')
    self._global_params = global_params
    self._blocks_args = blocks_args
    self._relu_fn = global_params.relu_fn
    self._batch_norm = global_params.batch_norm

    self._build()

  def _build(self):
    """Builds a model."""
    self._blocks = []
    batch_norm_momentum = self._global_params.batch_norm_momentum
    batch_norm_epsilon = self._global_params.batch_norm_epsilon
    if self._global_params.data_format == 'channels_first':
      channel_axis = 1
      self._spatial_dims = [2, 3]
    else:
      channel_axis = -1
      self._spatial_dims = [1, 2]

    self._conv_stem = tf.keras.layers.Conv2D(
        filters=round_filters(32, self._global_params),
        kernel_size=[3, 3],
        strides=[2, 2],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        data_format=self._global_params.data_format,
        use_bias=False)
    self._bn0 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)

    for block_args in self._blocks_args:
      assert block_args.num_repeat > 0
      # Update block input and output filters based on depth multiplier.
      input_filters = round_filters(block_args.input_filters,
                                    self._global_params)
      output_filters = round_filters(block_args.output_filters,
                                     self._global_params)
      repeats = round_repeats(block_args.num_repeat, self._global_params)
      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      self._blocks.append(MBConvBlock(block_args, self._global_params))

      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        self._blocks.append(MBConvBlock(block_args, self._global_params))

    self._conv_head = tf.keras.layers.Conv2D(
        filters=round_filters(1280, self._global_params),
        kernel_size=[1, 1],
        strides=[1, 1],
        kernel_initializer=conv_kernel_initializer,
        padding='same',
        use_bias=False)
    self._bn1 = self._batch_norm(
        axis=channel_axis,
        momentum=batch_norm_momentum,
        epsilon=batch_norm_epsilon)

    self._avg_pooling = tf.keras.layers.GlobalAveragePooling2D(
        data_format=self._global_params.data_format)
    if self._global_params.dropout_rate > 0:
      self._dropout = tf.keras.layers.Dropout(self._global_params.dropout_rate)
    else:
      self._dropout = None
    self._fc = tf.keras.layers.Dense(
        self._global_params.num_classes,
        kernel_initializer=dense_kernel_initializer)

  def call(self, inputs, training=True):
    """Implementation of call().

    Args:
      inputs: input tensors.
      training: boolean, whether the model is constructed for training.

    Returns:
      output tensors.
    """
    outputs = self._relu_fn(
        self._bn0(self._conv_stem(inputs), training=training))

    for idx, block in enumerate(self._blocks):
      survival_prob = self._global_params.survival_prob
      if survival_prob:
        drop_rate = 1.0 - survival_prob
        survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
      outputs = block.call(
          outputs, training=training, survival_prob=survival_prob)

    outputs = self._relu_fn(
        self._bn1(self._conv_head(outputs), training=training))
    outputs = self._avg_pooling(outputs)
    if self._dropout:
      outputs = self._dropout(outputs, training=training)
    outputs = self._fc(outputs)
    return outputs
