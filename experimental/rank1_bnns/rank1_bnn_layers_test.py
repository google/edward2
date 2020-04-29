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

# Lint as: python3
"""Tests for rank-1 BNN layers."""
import itertools

from absl.testing import parameterized
from experimental.rank1_bnns import rank1_bnn_layers  # local file import
import numpy as np
import tensorflow.compat.v2 as tf


class Rank1PriorLayersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'bias_initializer': 'zeros'},
      {'alpha_initializer': 'trainable_deterministic',
       'gamma_initializer': 'trainable_deterministic',
       'bias_initializer': 'trainable_deterministic'},
  )
  def testDenseRank1BatchEnsemble(self,
                                  alpha_initializer,
                                  gamma_initializer,
                                  bias_initializer):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1])
    layer = rank1_bnn_layers.DenseRank1(
        output_dim,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        bias_initializer=bias_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        activation=None,
        ensemble_size=ensemble_size)

    output = layer(batched_inputs)
    manual_output = [
        layer.dense(inputs*layer.alpha[i]) * layer.gamma[i] + layer.bias[i]
        for i in range(ensemble_size)]
    manual_output = tf.concat(manual_output, axis=0)

    expected_shape = (ensemble_size*examples_per_model, output_dim)
    self.assertEqual(output.shape, expected_shape)
    self.assertAllClose(output, manual_output)

  @parameterized.parameters(
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'trainable_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'trainable_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
  )
  def testDenseRank1AlphaGamma(self,
                               alpha_initializer,
                               gamma_initializer,
                               all_close,
                               use_additive_perturbation,
                               ensemble_size):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5*ensemble_size, 12).astype(np.float32)
    model = rank1_bnn_layers.DenseRank1(
        4,
        ensemble_size=ensemble_size,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        activation=None)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5*ensemble_size, 4))
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  def testDenseRank1Model(self):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding='SAME',
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        rank1_bnn_layers.DenseRank1(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    self.assertLen(model.losses, 2)

  @parameterized.parameters(
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal'},
      {'alpha_initializer': 'trainable_deterministic',
       'gamma_initializer': 'trainable_deterministic'},
  )
  def testConv2DRank1BatchEnsemble(self, alpha_initializer, gamma_initializer):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, 4, 4, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1, 1, 1])
    layer = rank1_bnn_layers.Conv2DRank1(
        output_dim,
        kernel_size=2,
        padding='same',
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        activation=None,
        ensemble_size=ensemble_size)

    output = layer(batched_inputs)
    manual_output = [
        layer.conv2d(inputs*layer.alpha[i]) * layer.gamma[i] + layer.bias[i]
        for i in range(ensemble_size)]
    manual_output = tf.concat(manual_output, axis=0)
    self.assertEqual(output.shape,
                     (ensemble_size*examples_per_model, 4, 4, output_dim))
    self.assertAllClose(output, manual_output)

  @parameterized.parameters(
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'he_normal',
       'gamma_initializer': 'he_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'zeros',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'trainable_normal',
       'all_close': True,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'zeros',
       'gamma_initializer': 'trainable_normal',
       'all_close': True,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': False,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': True,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': False,
       'ensemble_size': 4},
      {'alpha_initializer': 'trainable_normal',
       'gamma_initializer': 'trainable_normal',
       'all_close': False,
       'use_additive_perturbation': True,
       'ensemble_size': 4},
  )
  def testConv2DRank1AlphaGamma(self,
                                alpha_initializer,
                                gamma_initializer,
                                all_close,
                                use_additive_perturbation,
                                ensemble_size):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5*ensemble_size, 4, 4, 12).astype(np.float32)
    model = rank1_bnn_layers.Conv2DRank1(
        4,
        kernel_size=2,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        activation=None)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5*ensemble_size, 3, 3, 4))
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  def testConv2DRank1Model(self):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        rank1_bnn_layers.Conv2DRank1(3,
                                     kernel_size=2,
                                     padding='SAME',
                                     activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    self.assertLen(model.losses, 2)

  @parameterized.parameters(
      itertools.chain(
          itertools.product(
              ('he_normal',), ('he_normal',), ('he_normal',), ('he_normal',),
              ('he_normal',), (True, False), (1, 2), (True, False)),
          itertools.product(
              ('trainable_deterministic',), ('trainable_deterministic',),
              ('trainable_deterministic',), ('trainable_deterministic',),
              ('trainable_deterministic',), (True, False), (1, 2),
              (True, False)))
  )
  def testLSTMCellRank1BatchEnsemble(self, alpha_initializer, gamma_initializer,
                                     recurrent_alpha_initializer,
                                     recurrent_gamma_initializer,
                                     bias_initializer, use_bias, implementation,
                                     use_additive_perturbation):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 4
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1])
    layer = rank1_bnn_layers.LSTMCellRank1(
        output_dim,
        use_bias=use_bias,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        recurrent_alpha_initializer=recurrent_alpha_initializer,
        recurrent_gamma_initializer=recurrent_gamma_initializer,
        bias_initializer=bias_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        recurrent_alpha_regularizer=None,
        recurrent_gamma_regularizer=None,
        implementation=implementation,
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)
    h0 = tf.random.normal([examples_per_model, output_dim])
    c0 = tf.random.normal([examples_per_model, output_dim])

    def compute_rank1_lstm_cell(i):
      if use_additive_perturbation:
        ifgo = tf.linalg.matmul(
            inputs + layer.alpha[i], layer.kernel) + layer.gamma[i]
        ifgo += tf.linalg.matmul(
            h0 + layer.recurrent_alpha[i],
            layer.recurrent_kernel) + layer.recurrent_gamma[i]
      else:
        ifgo = tf.linalg.matmul(
            inputs * layer.alpha[i], layer.kernel) * layer.gamma[i]
        ifgo += tf.linalg.matmul(
            h0 * layer.recurrent_alpha[i],
            layer.recurrent_kernel) * layer.recurrent_gamma[i]
      if use_bias:
        ifgo += layer.bias[i]
      i, f, g, o = tf.split(ifgo, num_or_size_splits=4, axis=1)
      i = tf.nn.sigmoid(i)
      f = tf.nn.sigmoid(f)
      g = tf.nn.tanh(g)
      o = tf.nn.sigmoid(o)
      c = f*c0 + i*g
      h = o * tf.nn.tanh(c)
      return h

    h0_batched = tf.tile(h0, [ensemble_size, 1])
    c0_batched = tf.tile(c0, [ensemble_size, 1])
    outputs, _ = layer(batched_inputs, (h0_batched, c0_batched))
    manual_outputs = tf.concat(
        [compute_rank1_lstm_cell(i) for i in range(ensemble_size)], axis=0)

    expected_shape = (ensemble_size*examples_per_model, output_dim)
    self.assertEqual(outputs.shape, expected_shape)
    self.assertAllClose(outputs, manual_outputs)

    layer2 = rank1_bnn_layers.LSTMCellRank1.from_config(layer.get_config())
    layer2(batched_inputs, (h0_batched, c0_batched))  # force initialization
    layer2.set_weights(layer.get_weights())
    outputs2, _ = layer2(batched_inputs, (h0_batched, c0_batched))
    self.assertAllClose(outputs, outputs2)

  @parameterized.parameters(
      list(itertools.product(
          ('he_normal', 'trainable_normal',),
          ('he_normal', 'trainable_normal',),
          ('he_normal', 'trainable_normal',),
          ('he_normal', 'trainable_normal',),
          (1, 2), (True, False)))
  )
  def testLSTMCellRank1AlphaGamma(self, alpha_initializer, gamma_initializer,
                                  recurrent_alpha_initializer,
                                  recurrent_gamma_initializer,
                                  implementation, use_additive_perturbation):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 4
    batch_size = 5 * ensemble_size
    output_dim = 4
    inputs = np.random.rand(batch_size, 12).astype(np.float32)
    layer = rank1_bnn_layers.LSTMCellRank1(
        output_dim,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        recurrent_alpha_initializer=recurrent_alpha_initializer,
        recurrent_gamma_initializer=recurrent_gamma_initializer,
        ensemble_size=ensemble_size)
    h0 = tf.random.normal([batch_size, output_dim])
    c0 = tf.random.normal([batch_size, output_dim])
    outputs1, _ = layer(inputs, (h0, c0))
    layer._sample_weights(inputs)
    outputs2, _ = layer(inputs, (h0, c0))
    self.assertEqual(outputs1.shape, (batch_size, output_dim))
    all_close = 'trainable_normal' not in [alpha_initializer, gamma_initializer,
                                           recurrent_alpha_initializer,
                                           recurrent_gamma_initializer]
    if all_close:
      self.assertAllClose(outputs1, outputs2, rtol=1e-4)
    else:
      self.assertNotAllClose(outputs1, outputs2)

  @parameterized.parameters(
      list(itertools.product((1, 4), (1, 2), (True, False), (True, False)))
  )
  def testLSTMCellRank1Model(self, ensemble_size, implementation,
                             use_additive_perturbation, use_bias):
    batch_size = 2 * ensemble_size
    timesteps = 3
    input_dim = 12
    hidden_size = 10
    inputs = np.random.rand(batch_size, timesteps, input_dim).astype(np.float32)
    cell = rank1_bnn_layers.LSTMCellRank1(
        hidden_size, use_bias=use_bias, implementation=implementation,
        use_additive_perturbation=use_additive_perturbation,
        ensemble_size=ensemble_size)
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
    # NOTE: `cell.sample_weights` should have been called at the beginning of
    # each call, so these should be different.
    self.assertNotAllClose(outputs1, outputs2)
    # NOTE: We didn't call `cell.sample_weights` again before computing
    # `outputs3`, so the cell should have had the same weights as it did
    # during computation of `outputs2`, and thus yielded the same output
    # tensor.
    self.assertAllClose(outputs2, outputs3)
    self.assertLen(model.losses, 4)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
