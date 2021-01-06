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

"""Tests for Bayesian linear models."""

import edward2 as ed
import numpy as np
import tensorflow as tf


class BayesianLinearModelTest(tf.test.TestCase):

  def testBayesianLinearModel(self):
    """Tests that model makes reasonable predictions."""
    np.random.seed(42)
    train_batch_size = 5
    test_batch_size = 2
    num_features = 3
    noise_variance = 0.01
    coeffs = tf.range(num_features, dtype=tf.float32)
    features = tf.cast(np.random.randn(train_batch_size, num_features),
                       dtype=tf.float32)
    noise = tf.cast(np.random.randn(train_batch_size), dtype=tf.float32)
    labels = (tf.tensordot(features, coeffs, [[-1], [0]])
              + noise_variance * noise)

    model = ed.layers.BayesianLinearModel(noise_variance=noise_variance)
    model.fit(features, labels)

    test_features = np.random.randn(test_batch_size, num_features).astype(
        np.float32)
    test_labels = tf.tensordot(test_features, coeffs, [[-1], [0]])
    outputs = model(test_features)
    test_predictions = outputs.distribution.mean()
    test_predictions_variance = outputs.distribution.variance()

    self.assertEqual(test_predictions.shape, (test_batch_size,))
    self.assertEqual(test_predictions_variance.shape, (test_batch_size,))
    self.assertAllClose(test_predictions, test_labels, atol=0.1)
    self.assertAllLessEqual(test_predictions_variance, noise_variance)


if __name__ == '__main__':
  tf.test.main()
