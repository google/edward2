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

"""Routing layer for mixture of experts."""

import tensorflow as tf
from edward2.tensorflow.layers import routing_utils


class RoutingLayer(tf.keras.layers.Layer):

  def __init__(self, num_experts, routing_pooling, routing_fn, k,
               normalize_routing, noise_epsilon, **kwargs):
    super().__init__(**kwargs)
    self.num_experts = num_experts
    self.routing_pooling = routing_pooling
    self.routing_fn = routing_fn
    self.k = k
    self.normalize_routing = normalize_routing
    self.noise_epsilon = noise_epsilon
    self.use_noisy_routing = 'noisy' in routing_fn
    self.use_softmax_top_k = routing_fn in [
        'softmax_top_k', 'noisy_softmax_top_k'
    ]
    self.use_onehot_top_k = routing_fn in ['onehot_top_k', 'noisy_onehot_top_k']
    self.use_sigmoid_activation = routing_fn == 'sigmoid'
    self.use_softmax_routing = routing_fn in ['softmax', 'noisy_softmax']

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.input_size = input_shape[1]
    self.kernel_shape = [self.input_size, self.num_experts]

    self.w_gate = self.add_weight(
        name='w_gate',
        shape=self.kernel_shape,
        initializer=tf.keras.initializers.Zeros(),
        regularizer=None,
        constraint=None,
        trainable=True,
        dtype=self.dtype)

    if self.use_noisy_routing:
      self.w_noise = self.add_weight(
          name='w_gate',
          shape=self.kernel_shape,
          initializer=tf.keras.initializers.Zeros(),
          regularizer=None,
          constraint=None,
          trainable=True,
          dtype=self.dtype)

    if self.routing_pooling == 'global_average':
      self.pooling_layer = tf.keras.layers.GlobalAveragePooling2D()
    elif self.routing_pooling == 'global_max':
      self.pooling_layer = tf.keras.layers.GlobalMaxPool2D()
    elif self.routing_pooling == 'average_8':
      self.pooling_layer = tf.keras.Sequential([
          tf.keras.layers.AveragePooling2D(pool_size=8),
          tf.keras.layers.Flatten(),
      ])
    elif self.routing_pooling == 'max_8':
      self.pooling_layer = tf.keras.Sequential([
          tf.keras.layers.MaxPool2D(pool_size=8),
          tf.keras.layers.Flatten(),
      ])
    else:
      self.pooling_layer = tf.keras.layers.Flatten()

    self.built = True

  def call(self, inputs, training=None):
    pooled_inputs = self.pooling_layer(inputs)
    routing_weights = tf.linalg.matmul(pooled_inputs, self.w_gate)

    if self.use_noisy_routing and training:
      raw_noise_stddev = tf.linalg.matmul(pooled_inputs, self.w_noise)
      noise_stddev = tf.nn.softplus(raw_noise_stddev) + self.noise_epsilon
      routing_weights += tf.random.normal(
          tf.shape(routing_weights)) * noise_stddev

    if self.use_sigmoid_activation:
      routing_weights = tf.nn.sigmoid(routing_weights)
    elif self.use_softmax_routing:
      routing_weights = tf.nn.softmax(routing_weights)
    elif self.use_softmax_top_k:
      top_values, top_indices = tf.math.top_k(routing_weights,
                                              min(self.k + 1, self.num_experts))
      # top k logits has shape [batch, k]
      top_k_values = tf.slice(top_values, [0, 0], [-1, self.k])
      top_k_indices = tf.slice(top_indices, [0, 0], [-1, self.k])
      top_k_gates = tf.nn.softmax(top_k_values)
      # This returns a [batch, n] Tensor with 0's in the positions of non-top-k
      # expert values.
      routing_weights = routing_utils.rowwise_unsorted_segment_sum(
          top_k_gates, top_k_indices, self.num_experts)
    elif self.use_onehot_top_k:
      top_values, top_indices = tf.math.top_k(routing_weights, k=self.k)
      one_hot_tensor = tf.one_hot(top_indices, depth=self.num_experts)
      mask = tf.reduce_sum(one_hot_tensor, axis=1)
      routing_weights *= mask

    if self.normalize_routing:
      normalization = tf.math.reduce_sum(
          routing_weights, axis=-1, keepdims=True)
      routing_weights /= normalization

    return routing_weights

    def get_config(self):
      config = {
          'num_experts': self.num_experts,
          'routing_pooling': self.routing_pooling,
          'routing_fn': self.routing_fn,
          'k': self.k,
          'normalize_routing': self.normalize_routing,
          'noise_epsilon': self.noise_epsilon,
      }
      new_config = super().get_config()
      new_config.update(config)
      return new_config
