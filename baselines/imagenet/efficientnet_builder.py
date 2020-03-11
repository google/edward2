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

import os
import re
from absl import logging
import efficientnet_model  # local file import
import numpy as np
import six
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


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    if six.PY2:
      assert isinstance(block_string, (str, unicode))
    else:
      assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return efficientnet_model.BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]),
                 int(options['s'][1])],
        conv_type=int(options['c']) if 'c' in options else 0,
        fused_conv=int(options['f']) if 'f' in options else 0,
        super_pixel=int(options['p']) if 'p' in options else 0,
        condconv=('cc' in block_string))

  def _encode_block_string(self, block):
    """Encodes a block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters,
        'c%d' % block.conv_type,
        'f%d' % block.fused_conv,
        'p%d' % block.super_pixel,
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:  # pylint: disable=g-bool-id-comparison
      args.append('noskip')
    if block.condconv:
      args.append('cc')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of block.

    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent blocks arguments.
    Returns:
      a list of strings, each string is a notation of block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


def swish(features, use_native=True, use_hard=False):
  """Computes the Swish activation function.

  We provide three alternnatives:
    - Native tf.nn.swish, use less memory during training than composable swish.
    - Quantization friendly hard swish.
    - A composable swish, equivalant to tf.nn.swish, but more general for
      finetuning and TF-Hub.

  Args:
    features: A `Tensor` representing preactivation values.
    use_native: Whether to use the native swish from tf.nn that uses a custom
      gradient to reduce memory usage, or to use customized swish that uses
      default TensorFlow gradient computation.
    use_hard: Whether to use quantization-friendly hard swish.

  Returns:
    The activation value.
  """
  if use_native and use_hard:
    raise ValueError('Cannot specify both use_native and use_hard.')

  if use_native:
    return tf.nn.swish(features)

  if use_hard:
    return features * tf.nn.relu6(features + np.float32(3)) * (1. / 6.)

  features = tf.convert_to_tensor(features, name='features')
  return features * tf.nn.sigmoid(features)


_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]


def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 survival_prob=0.8):
  """Creates a efficientnet model."""
  global_params = efficientnet_model.GlobalParams(
      blocks_args=_DEFAULT_BLOCKS_ARGS,
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      survival_prob=survival_prob,
      data_format='channels_last',
      num_classes=1000,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None,
      relu_fn=tf.nn.swish,
      # The default is TPU-specific batch norm.
      # The alternative is tf.layers.BatchNormalization.
      batch_norm=SyncBatchNorm,  # TPU-specific requirement.
      use_se=True,
      clip_projection_output=False)
  return global_params


def get_model_params(model_name, override_params):
  """Get the block args and global params for a given model."""
  if model_name.startswith('efficientnet'):
    width_coefficient, depth_coefficient, _, dropout_rate = (
        efficientnet_params(model_name))
    global_params = efficientnet(
        width_coefficient, depth_coefficient, dropout_rate)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  decoder = BlockDecoder()
  blocks_args = decoder.decode(global_params.blocks_args)

  logging.info('global_params= %s', global_params)
  return blocks_args, global_params


def build_model(model_name,
                override_params=None,
                model_dir=None):
  """A helper function to create a model and return predicted logits.

  Args:
    model_name: string, the predefined model name.
    override_params: A dictionary of params for overriding. Fields must exist in
      efficientnet_model.GlobalParams.
    model_dir: string, optional model dir for saving configs.

  Returns:
    logits: the logits tensor of classes.
  """
  # For backward compatibility.
  if override_params and override_params.get('drop_connect_rate', None):
    override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

  blocks_args, global_params = get_model_params(model_name, override_params)

  if model_dir:
    param_file = os.path.join(model_dir, 'model_params.txt')
    if not tf.gfile.Exists(param_file):
      if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
      with tf.gfile.GFile(param_file, 'w') as f:
        logging.info('writing to %s', param_file)
        f.write('model_name= %s\n\n' % model_name)
        f.write('global_params= %s\n\n' % str(global_params))
        f.write('blocks_args= %s\n\n' % str(blocks_args))

  return efficientnet_model.Model(blocks_args, global_params)
