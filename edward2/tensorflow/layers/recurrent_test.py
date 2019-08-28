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

"""Tests for Bayesian recurrent cells and layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf

tfe = tf1.contrib.eager


class RecurrentTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testLSTMCell(self,
                   lstm_cell,
                   kernel_initializer,
                   recurrent_initializer,
                   bias_initializer,
                   all_close):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = np.random.rand(batch_size, timesteps, dim).astype(np.float32)
    cell = lstm_cell(hidden_size,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer)
    noise = np.random.rand(1, hidden_size).astype(np.float32)
    h0, c0 = cell.get_initial_state(inputs)
    state = (h0 + noise, c0)
    outputs1, _ = cell(inputs[:, 0, :], state)
    outputs2, _ = cell(inputs[:, 0, :], state)
    cell.call_weights()
    outputs3, _ = cell(inputs[:, 0, :], state)
    self.evaluate(tf1.global_variables_initializer())
    res1, res2, res3 = self.evaluate([outputs1, outputs2, outputs3])
    self.assertEqual(res1.shape, (batch_size, hidden_size))
    self.assertAllClose(res1, res2)
    if all_close:
      self.assertAllClose(res1, res3)
    else:
      self.assertNotAllClose(res1, res3)
    cell.get_config()

  @parameterized.parameters(
      {"lstm_cell": ed.layers.LSTMCellFlipout},
      {"lstm_cell": ed.layers.LSTMCellReparameterization},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testLSTMCellLoss(self, lstm_cell):
    features = np.random.rand(5, 1, 12).astype(np.float32)
    labels = np.random.rand(5, 10).astype(np.float32)
    cell = lstm_cell(10)
    state = (tf.zeros([1, 10]), tf.zeros([1, 10]))

    # Imagine this is the 1st epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions, _ = cell(features[:, 0, :], state)  # first call forces build
      cell(features[:, 0, :], state)  # ensure robustness after multiple calls
      cell.get_initial_state(features[:, 0, :])
      cell(features[:, 0, :], state)  # ensure robustness after multiple calls
      nll = tf.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(cell.losses)

    variables = [
        cell.kernel_initializer.mean, cell.kernel_initializer.stddev,
        cell.recurrent_initializer.mean, cell.recurrent_initializer.stddev,
    ]
    for v in variables:
      self.assertIn(v, cell.variables)

    # This will be fine, since the layer was built inside this tape, and thus
    # the distribution init ops were inside this tape.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    # Imagine this is the 2nd epoch.
    with tf.GradientTape(persistent=True) as tape:
      cell.get_initial_state(features[:, 0, :])
      predictions, _ = cell(features[:, 0, :], state)  # build is not called
      nll = tf.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(cell.losses)

    variables = [
        cell.kernel_initializer.mean, cell.kernel_initializer.stddev,
        cell.recurrent_initializer.mean, cell.recurrent_initializer.stddev,
    ]
    for v in variables:
      self.assertIn(v, cell.variables)

    # This would fail, since the layer was built inside the tape from the 1st
    # epoch, and thus the distribution init ops were inside that tape instead of
    # this tape. By using a callable for the variable, this will no longer fail.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

  @parameterized.parameters(
      {"lstm_cell": ed.layers.LSTMCellFlipout},
      {"lstm_cell": ed.layers.LSTMCellReparameterization},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testLSTMCellModel(self, lstm_cell):
    batch_size, timesteps, dim = 5, 3, 12
    hidden_size = 10
    inputs = np.random.rand(batch_size, timesteps, dim).astype(np.float32)
    cell = lstm_cell(hidden_size)
    model = tf.keras.Sequential([
        tf.keras.layers.RNN(cell, return_sequences=True)
    ])
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    state = (tf.zeros([1, hidden_size]), tf.zeros([1, hidden_size]))
    outputs3 = []
    for t in range(timesteps):
      out, state = cell(inputs[:, t, :], state)
      outputs3.append(out)
    outputs3 = tf.stack(outputs3, axis=1)
    self.evaluate(tf1.global_variables_initializer())
    res1, res2, res3 = self.evaluate([outputs1, outputs2, outputs3])
    self.assertEqual(res1.shape, (batch_size, timesteps, hidden_size))
    self.assertEqual(res3.shape, (batch_size, timesteps, hidden_size))
    # NOTE: `cell.call_weights` should have been called at the beginning of
    # each call, so these should be different.
    self.assertNotAllClose(res1, res2)
    # NOTE: We didn't call `cell.call_weights` again before computing
    # `outputs3`, so the cell should have had the same weights as it did
    # during computation of `outputs2`, and thus yielded the same output
    # tensor.
    self.assertAllClose(res2, res3)
    self.assertLen(model.losses, 2)


if __name__ == "__main__":
  tf.test.main()
