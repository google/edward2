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

"""Tests for normalization layers."""

import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf


class NormalizationTest(tf.test.TestCase):

  def testActNorm(self):
    np.random.seed(83243)
    batch_size = 25
    length = 15
    channels = 4
    inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    inputs = tf.cast(inputs, tf.float32)
    layer = ed.layers.ActNorm()
    outputs = layer(inputs)
    mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    self.assertAllClose(mean, np.zeros(channels), atol=1e-3)
    self.assertAllClose(variance, np.ones(channels), atol=1e-3)

    inputs = 3. + 0.8 * np.random.randn(batch_size, length, channels)
    inputs = tf.cast(inputs, tf.float32)
    outputs = layer(inputs)
    mean, variance = tf.nn.moments(outputs, axes=[0, 1])
    self.assertAllClose(mean, np.zeros(channels), atol=0.25)
    self.assertAllClose(variance, np.ones(channels), atol=0.25)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
