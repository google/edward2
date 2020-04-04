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

"""Tests for Bayesian convolutional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf


class ConvolutionalTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  def testConv2DKernel(self,
                       layer,
                       kernel_initializer,
                       bias_initializer,
                       all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5, 4, 4, 12).astype(np.float32)
    model = layer(4,
                  kernel_size=2,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation=tf.nn.relu)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5, 3, 3, 4))
    self.assertAllGreaterEqual(outputs1, 0.)
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  @parameterized.parameters(
      {"layer": ed.layers.Conv2DFlipout},
      {"layer": ed.layers.Conv2DHierarchical},
      {"layer": ed.layers.Conv2DReparameterization},
      {"layer": ed.layers.Conv2DVariationalDropout},
  )
  def testConv2DModel(self, layer):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        layer(3, kernel_size=2, padding="SAME", activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    if layer == ed.layers.Conv2DHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @parameterized.parameters(
      {"layer_cls": ed.layers.Conv2DBatchEnsemble},
      {"layer_cls": ed.layers.DepthwiseConv2DBatchEnsemble},
  )
  def testConv2DBatchEnsemble(self, layer_cls):
    """Tests that vectorized implementation is same as for loop."""
    ensemble_size = 2
    examples_per_model = 3
    channels = 5
    inputs = tf.random.normal([examples_per_model, 4, 4, channels])
    if layer_cls == ed.layers.Conv2DBatchEnsemble:
      layer_cls = functools.partial(layer_cls, filters=channels)
    layer = layer_cls(
        kernel_size=2,
        ensemble_size=ensemble_size,
        activation=None)

    batch_inputs = tf.tile(inputs, [ensemble_size, 1, 1, 1])
    batch_outputs = layer(batch_inputs)
    loop_outputs = [
        layer.conv2d(inputs*layer.alpha[i]) * layer.gamma[i] + layer.bias[i]
        for i in range(ensemble_size)]
    loop_outputs = tf.concat(loop_outputs, axis=0)

    expected_shape = (ensemble_size * examples_per_model, 3, 3, channels)
    self.assertEqual(batch_outputs.shape, expected_shape)
    self.assertAllClose(batch_outputs, loop_outputs)

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
