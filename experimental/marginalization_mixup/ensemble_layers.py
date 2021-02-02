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

"""Implement ensemble layers.
"""

import tensorflow as tf


class BatchEnsembleDEConv2D(tf.python.keras.layers.Layer):
  """A batch ensemble convolutional transpose layer."""

  def __init__(self,
               filters,
               kernel_size,
               num_models=4,
               alpha_initializer=tf.python.keras.initializers.Ones(),
               gamma_initializer=tf.python.keras.initializers.Ones(),
               strides=(1, 1),
               padding="valid",
               data_format="channels_last",
               activation=None,
               use_bias=True,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(BatchEnsembleDEConv2D, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.data_format = data_format
    self.num_models = num_models
    self.alpha_initializer = alpha_initializer
    self.gamma_initializer = gamma_initializer
    self.use_bias = use_bias
    self.activation = tf.python.keras.activations.get(activation)
    self.deconv2d = tf.python.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == "channels_first":
      input_channel = input_shape[1]
    elif self.data_format == "channels_last":
      input_channel = input_shape[-1]

    self.alpha = self.add_weight(
        "alpha",
        shape=[self.num_models, input_channel],
        initializer=self.alpha_initializer,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        "gamma",
        shape=[self.num_models, self.filters],
        initializer=self.gamma_initializer,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name="bias",
          shape=[self.num_models, self.filters],
          initializer=tf.python.keras.initializers.Zeros(),
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    axis_change = -1 if self.data_format == "channels_first" else 1
    batch_size = tf.shape(inputs)[0]
    examples_per_model = batch_size // self.num_models
    alpha = tf.reshape(
        tf.tile(self.alpha, [1, examples_per_model]), [batch_size, -1])
    gamma = tf.reshape(
        tf.tile(self.gamma, [1, examples_per_model]), [batch_size, -1])
    alpha = tf.expand_dims(alpha, axis=axis_change)
    alpha = tf.expand_dims(alpha, axis=axis_change)
    gamma = tf.expand_dims(gamma, axis=axis_change)
    gamma = tf.expand_dims(gamma, axis=axis_change)
    outputs = self.deconv2d(inputs*alpha) * gamma
    if self.use_bias:
      bias = tf.reshape(
          tf.tile(self.bias, [1, examples_per_model]), [batch_size, -1])
      bias = tf.expand_dims(bias, axis=axis_change)
      bias = tf.expand_dims(bias, axis=axis_change)
      outputs += bias
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs
