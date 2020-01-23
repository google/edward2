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

"""Tests for stochastic output layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class StochasticOutputTest(parameterized.TestCase, tf.test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def testMixtureLogistic(self):
    batch_size = 3
    features = np.random.rand(batch_size, 4).astype(np.float32)
    labels = np.random.rand(batch_size).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=None),
        ed.layers.MixtureLogistic(5),
    ])
    outputs = model(features)
    log_likelihood = tf.reduce_sum(outputs.distribution.log_prob(labels))
    self.evaluate(tf1.global_variables_initializer())
    log_likelihood_val, outputs_val = self.evaluate([log_likelihood, outputs])
    self.assertEqual(log_likelihood_val.shape, ())
    self.assertLessEqual(log_likelihood_val, 0.)
    self.assertEqual(outputs_val.shape, (batch_size,))


if __name__ == "__main__":
  tf.test.main()
