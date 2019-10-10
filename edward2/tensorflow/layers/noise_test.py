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

"""Tests for Bayesian noise layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class NoiseTest(parameterized.TestCase, tf.test.TestCase):

  def testNCPNormalPerturb(self):
    batch_size = 3
    inputs = tf.cast(np.random.rand(batch_size, 4), dtype=tf.float32)
    model = ed.layers.NCPNormalPerturb()
    outputs = model(inputs)
    inputs_val, outputs_val = self.evaluate([inputs, outputs])
    self.assertEqual(outputs_val.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs_val, outputs_val[:batch_size])

  def testNCPCategoricalPerturb(self):
    input_dim = 5
    batch_size = 3
    inputs = tf.cast(np.random.choice(input_dim, size=(batch_size, 4)),
                     dtype=tf.float32)
    model = ed.layers.NCPCategoricalPerturb(input_dim)
    outputs = model(inputs)
    inputs_val, outputs_val = self.evaluate([inputs, outputs])
    self.assertEqual(outputs_val.shape, (2 * batch_size, 4))
    self.assertAllEqual(inputs_val, outputs_val[:batch_size])

  def testNCPNormalOutput(self):
    batch_size = 3
    features = ed.Normal(loc=tf.random.normal([2 * batch_size, 1]), scale=1.)
    labels = np.random.rand(batch_size).astype(np.float32)
    model = ed.layers.NCPNormalOutput(mean=labels)
    predictions = model(features)
    features_val, predictions_val = self.evaluate([features, predictions])
    self.assertLen(model.losses, 1)
    self.assertAllEqual(features_val[:batch_size], predictions_val)

if __name__ == "__main__":
  tf.test.main()
