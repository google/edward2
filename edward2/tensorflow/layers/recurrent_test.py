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

"""Tests for Bayesian recurrent cells and layers."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf


class RecurrentTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 1,
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 1,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "implementation": 1,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 2,
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 2,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellFlipout,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "implementation": 2,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 1,
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 1,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "implementation": 1,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "zeros",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "trainable_normal",
       "implementation": 1,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "trainable_normal",
       "recurrent_initializer": "orthogonal",
       "bias_initializer": "zeros",
       "implementation": 2,
       "all_close": False},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "trainable_deterministic",
       "recurrent_initializer": "trainable_deterministic",
       "bias_initializer": "trainable_deterministic",
       "implementation": 1,
       "all_close": True},
      {"lstm_cell": ed.layers.LSTMCellReparameterization,
       "kernel_initializer": "trainable_deterministic",
       "recurrent_initializer": "trainable_deterministic",
       "bias_initializer": "trainable_deterministic",
       "implementation": 2,
       "all_close": True},
  )
  def testLSTMCell(self,
                   lstm_cell,
                   kernel_initializer,
                   recurrent_initializer,
                   bias_initializer,
                   implementation,
                   all_close):
    batch_size, dim = 5, 12
    hidden_size = 10
    inputs = np.random.rand(batch_size, dim).astype(np.float32)
    cell = lstm_cell(hidden_size,
                     kernel_initializer=kernel_initializer,
                     recurrent_initializer=recurrent_initializer,
                     bias_initializer=bias_initializer,
                     implementation=implementation)
    noise = np.random.rand(1, hidden_size).astype(np.float32)
    h0, c0 = cell.get_initial_state(inputs)
    state = (h0 + noise, c0)
    outputs1, _ = cell(inputs, state)
    outputs2, _ = cell(inputs, state)
    cell.call_weights()
    outputs3, _ = cell(inputs, state)
    self.assertEqual(outputs1.shape, (batch_size, hidden_size))
    self.assertAllClose(outputs1, outputs2)
    if all_close:
      self.assertAllClose(outputs1, outputs3)
    else:
      self.assertNotAllClose(outputs1, outputs3)
    cell.get_config()

  @parameterized.parameters(
      {"lstm_cell": ed.layers.LSTMCellFlipout},
      {"lstm_cell": ed.layers.LSTMCellReparameterization},
  )
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
      # Note in TF 2.0, checking membership (v in cell.weights) raises an error
      # for lists of differently shaped Tensors.
      self.assertTrue(any(v is weight for weight in cell.weights))

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
      # Note in TF 2.0, checking membership (v in cell.weights) raises an error
      # for lists of differently shaped Tensors.
      self.assertTrue(any(v is weight for weight in cell.weights))

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
    self.assertEqual(outputs1.shape, (batch_size, timesteps, hidden_size))
    self.assertEqual(outputs3.shape, (batch_size, timesteps, hidden_size))
    # NOTE: `cell.call_weights` should have been called at the beginning of
    # each call, so these should be different.
    self.assertNotAllClose(outputs1, outputs2)
    # NOTE: We didn't call `cell.call_weights` again before computing
    # `outputs3`, so the cell should have had the same weights as it did
    # during computation of `outputs2`, and thus yielded the same output
    # tensor.
    self.assertAllClose(outputs2, outputs3)
    self.assertLen(model.losses, 2)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
