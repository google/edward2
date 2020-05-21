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

"""Tests for Bayesian noise layers."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class NoiseTest(parameterized.TestCase, tf.test.TestCase):

  def testNCPNormalPerturb(self):
    batch_size = 3
    inputs = tf.cast(np.random.rand(batch_size, 4), dtype=tf.float32)
    model = ed.layers.NCPNormalPerturb()
    outputs = model(inputs)
    self.assertEqual(outputs.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs, outputs[:batch_size])

  def testNCPCategoricalPerturb(self):
    input_dim = 5
    batch_size = 3
    inputs = tf.cast(np.random.choice(input_dim, size=(batch_size, 4)),
                     dtype=tf.float32)
    model = ed.layers.NCPCategoricalPerturb(input_dim)
    outputs = model(inputs)
    self.assertEqual(outputs.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs, outputs[:batch_size])

  def testNCPNormalOutput(self):
    batch_size = 3
    features = ed.Normal(loc=tf.random.normal([2 * batch_size, 1]), scale=1.)
    labels = np.random.rand(batch_size).astype(np.float32)
    model = ed.layers.NCPNormalOutput(mean=labels)
    predictions = model(features)
    self.assertLen(model.losses, 1)
    self.assertAllEqual(tf.convert_to_tensor(features[:batch_size]),
                        tf.convert_to_tensor(predictions))

if __name__ == "__main__":
  tf.test.main()
