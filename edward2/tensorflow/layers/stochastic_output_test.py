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

"""Tests for stochastic output layers."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class StochasticOutputTest(parameterized.TestCase, tf.test.TestCase):

  def testMixtureLogistic(self):
    batch_size = 3
    features = np.random.rand(batch_size, 4).astype(np.float32)
    labels = np.random.rand(batch_size).astype(np.float32)
    model = tf.python.keras.Sequential([
        tf.python.keras.layers.Dense(2, activation=None),
        ed.layers.MixtureLogistic(5),
    ])
    outputs = model(features)
    log_likelihood = tf.reduce_sum(outputs.distribution.log_prob(labels))
    self.assertEqual(log_likelihood.shape, ())
    self.assertLessEqual(log_likelihood, 0.)
    self.assertEqual(outputs.shape, (batch_size,))


if __name__ == "__main__":
  tf.test.main()
