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
"""Multihead layers."""
import tensorflow as tf


# TODO(trandustin): Move into ed.layers.
class DenseMultihead(tf.python.keras.layers.Dense):
  """Multiheaded output layer."""

  def __init__(self,
               units,
               ensemble_size=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        units=units * ensemble_size,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    self.ensemble_size = ensemble_size

  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    # NOTE: This restricts this layer from being called on tensors of ndim > 2.
    outputs = super().call(inputs)
    outputs = tf.reshape(outputs, [batch_size,
                                   self.ensemble_size,
                                   self.units // self.ensemble_size])
    return outputs

  def get_config(self):
    config = {
        'units': self.units // self.ensemble_size,
        'ensemble_size': self.ensemble_size,
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config
