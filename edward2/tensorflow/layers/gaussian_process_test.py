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

"""Tests for Gaussian process layers."""

import edward2 as ed
import numpy as np
import tensorflow as tf


class GaussianProcessTest(tf.test.TestCase):

  def testGaussianProcessPosterior(self):
    train_batch_size = 3
    test_batch_size = 2
    input_dim = 4
    output_dim = 5
    features = np.random.rand(train_batch_size, input_dim).astype(np.float32)
    labels = np.random.rand(train_batch_size, output_dim).astype(np.float32)
    layer = ed.layers.GaussianProcess(output_dim,
                                      conditional_inputs=features,
                                      conditional_outputs=labels)
    test_features = np.random.rand(test_batch_size, input_dim).astype(
        np.float32)
    test_labels = np.random.rand(test_batch_size, output_dim).astype(
        np.float32)
    test_outputs = layer(test_features)
    test_nats = -test_outputs.distribution.log_prob(test_labels)
    self.assertEqual(test_nats.shape, ())
    self.assertGreaterEqual(test_nats, 0.)
    self.assertEqual(test_outputs.shape, (test_batch_size, output_dim))

  def testGaussianProcessPrior(self):
    batch_size = 3
    input_dim = 4
    output_dim = 5
    features = np.random.rand(batch_size, input_dim).astype(np.float32)
    labels = np.random.rand(batch_size, output_dim).astype(np.float32)
    model = tf.python.keras.Sequential([
        tf.python.keras.layers.Dense(2, activation=None),
        ed.layers.GaussianProcess(output_dim),
    ])
    outputs = model(features)
    log_prob = outputs.distribution.log_prob(labels)
    self.assertEqual(log_prob.shape, ())
    self.assertLessEqual(log_prob, 0.)
    self.assertEqual(outputs.shape, (batch_size, output_dim))

  def testSparseGaussianProcess(self):
    dataset_size = 10
    batch_size = 3
    input_dim = 4
    output_dim = 5
    features = np.random.rand(batch_size, input_dim).astype(np.float32)
    labels = np.random.rand(batch_size, output_dim).astype(np.float32)
    model = ed.layers.SparseGaussianProcess(output_dim, num_inducing=2)
    with tf.GradientTape() as tape:
      predictions = model(features)
      nll = -tf.reduce_mean(predictions.distribution.log_prob(labels))
      kl = sum(model.losses) / dataset_size
      loss = nll + kl

    grads = tape.gradient(loss, model.variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    self.assertEqual(loss.shape, ())
    self.assertGreaterEqual(loss, 0.)
    self.assertEqual(predictions.shape, (batch_size, output_dim))

    # Check that gradients work on a second iteration. This can fail if
    # trainable initializers do not recall their weights.
    with tf.GradientTape() as tape:
      predictions = model(features)
      nll = -tf.reduce_mean(predictions.distribution.log_prob(labels))
      kl = sum(model.losses) / dataset_size
      loss = nll + kl

    grads = tape.gradient(loss, model.variables)
    for grad in grads:
      self.assertIsNotNone(grad)

if __name__ == '__main__':
  tf.test.main()
