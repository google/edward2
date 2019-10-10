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

"""Tests for Gaussian process layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
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
    self.evaluate(tf1.global_variables_initializer())
    test_nats_val, outputs_val = self.evaluate([test_nats, test_outputs])
    self.assertEqual(test_nats_val.shape, ())
    self.assertGreaterEqual(test_nats_val, 0.)
    self.assertEqual(outputs_val.shape, (test_batch_size, output_dim))

  def testGaussianProcessPrior(self):
    batch_size = 3
    input_dim = 4
    output_dim = 5
    features = np.random.rand(batch_size, input_dim).astype(np.float32)
    labels = np.random.rand(batch_size, output_dim).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2, activation=None),
        ed.layers.GaussianProcess(output_dim),
    ])
    outputs = model(features)
    log_prob = outputs.distribution.log_prob(labels)
    self.evaluate(tf1.global_variables_initializer())
    log_prob_val, outputs_val = self.evaluate([log_prob, outputs])
    self.assertEqual(log_prob_val.shape, ())
    self.assertLessEqual(log_prob_val, 0.)
    self.assertEqual(outputs_val.shape, (batch_size, output_dim))

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

    self.evaluate(tf1.global_variables_initializer())
    grads = tape.gradient(nll, model.variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    loss_val, predictions_val = self.evaluate([loss, predictions])
    self.assertEqual(loss_val.shape, ())
    self.assertGreaterEqual(loss_val, 0.)
    self.assertEqual(predictions_val.shape, (batch_size, output_dim))


if __name__ == '__main__':
  tf.test.main()
