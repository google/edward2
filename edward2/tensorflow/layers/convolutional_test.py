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

"""Tests for Bayesian convolutional layers."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class ConvolutionalTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DHierarchical,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv2DVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  def testConv2DKernel(self,
                       layer,
                       kernel_initializer,
                       bias_initializer,
                       all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5, 4, 4, 12).astype(np.float32)
    model = layer(4,
                  kernel_size=2,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation="relu")
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5, 3, 3, 4))
    self.assertAllGreaterEqual(outputs1, 0.)
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  @parameterized.parameters(
      {"layer": ed.layers.Conv2DFlipout},
      {"layer": ed.layers.Conv2DHierarchical},
      {"layer": ed.layers.Conv2DReparameterization},
      {"layer": ed.layers.Conv2DVariationalDropout},
      {"layer": ed.layers.Conv2DRank1},
  )
  def testConv2DModel(self, layer):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        layer(3, kernel_size=2, padding="SAME", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    if layer == ed.layers.Conv2DHierarchical:
      self.assertLen(model.losses, 3)
    elif layer == ed.layers.Conv2DRank1:
      self.assertLen(model.losses, 2)
    else:
      self.assertLen(model.losses, 1)

  @parameterized.parameters(
      {"layer": ed.layers.Conv1DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv1DFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv1DFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.Conv1DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.Conv1DReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.Conv1DReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  def testConv1DKernel(self, layer, kernel_initializer, bias_initializer,
                       all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5, 4, 12).astype(np.float32)
    model = layer(
        4,
        kernel_size=2,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        activation="relu")
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5, 3, 4))
    self.assertAllGreaterEqual(outputs1, 0.)
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  @parameterized.parameters(
      {"layer": ed.layers.Conv1DFlipout},
      {"layer": ed.layers.Conv1DReparameterization},
      {"layer": ed.layers.Conv1DRank1},
  )
  def testConv1DModel(self, layer):
    inputs = np.random.rand(3, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        layer(3, kernel_size=2, padding="SAME", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    if layer == ed.layers.Conv1DRank1:
      self.assertLen(model.losses, 2)
    else:
      self.assertLen(model.losses, 1)

  def testDepthwiseConv2DBatchEnsemble(self):
    """Tests that vectorized implementation is same as for loop."""
    ensemble_size = 2
    examples_per_model = 3
    channels = 5
    inputs = tf.random.normal([examples_per_model, 4, 4, channels])
    layer = ed.layers.DepthwiseConv2DBatchEnsemble(
        kernel_size=2,
        ensemble_size=ensemble_size,
        activation=None)

    batch_inputs = tf.tile(inputs, [ensemble_size, 1, 1, 1])
    batch_outputs = layer(batch_inputs)
    loop_outputs = [
        super(ed.layers.DepthwiseConv2DBatchEnsemble, layer).call(
            inputs * layer.alpha[i]) * layer.gamma[i] + layer.ensemble_bias[i]
        for i in range(ensemble_size)
    ]
    loop_outputs = tf.concat(loop_outputs, axis=0)

    expected_shape = (ensemble_size * examples_per_model, 3, 3, channels)
    self.assertEqual(batch_outputs.shape, expected_shape)
    self.assertAllClose(batch_outputs, loop_outputs)

  def testConv2DBatchEnsemble(self):
    """Tests that vectorized implementation is same as for loop."""
    ensemble_size = 2
    examples_per_model = 3
    channels = 5
    inputs = tf.random.normal([examples_per_model, 4, 4, channels])
    layer = ed.layers.Conv2DBatchEnsemble(
        filters=channels,
        kernel_size=2,
        ensemble_size=ensemble_size,
        activation=None)

    batch_inputs = tf.tile(inputs, [ensemble_size, 1, 1, 1])
    batch_outputs = layer(batch_inputs)

    loop_outputs = [
        super(ed.layers.Conv2DBatchEnsemble, layer).call(
            inputs * layer.alpha[i]) * layer.gamma[i] + layer.ensemble_bias[i]
        for i in range(ensemble_size)
    ]
    loop_outputs = tf.concat(loop_outputs, axis=0)

    expected_shape = (ensemble_size * examples_per_model, 3, 3, channels)
    self.assertEqual(batch_outputs.shape, expected_shape)
    self.assertAllClose(batch_outputs, loop_outputs)

  def testConv1DBatchEnsemble(self):
    """Tests that vectorized implementation is same as for loop."""
    ensemble_size = 2
    examples_per_model = 3
    channels = 5
    inputs = tf.random.normal([examples_per_model, 4, channels])
    layer = ed.layers.Conv1DBatchEnsemble(
        filters=channels,
        kernel_size=2,
        ensemble_size=ensemble_size,
        activation=None)
    batch_inputs = tf.tile(inputs, [ensemble_size, 1, 1])
    batch_outputs = layer(batch_inputs)
    loop_outputs = [
        super(ed.layers.Conv1DBatchEnsemble, layer).call(
            inputs * layer.alpha[i]) * layer.gamma[i] + layer.ensemble_bias[i]
        for i in range(ensemble_size)
    ]
    loop_outputs = tf.concat(loop_outputs, axis=0)

    expected_shape = (ensemble_size * examples_per_model, 3, channels)
    self.assertEqual(batch_outputs.shape, expected_shape)
    self.assertAllClose(batch_outputs, loop_outputs)

  @parameterized.parameters(
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal"},
      {"alpha_initializer": "trainable_deterministic",
       "gamma_initializer": "trainable_deterministic"},
  )
  def testConv2DRank1BatchEnsemble(self, alpha_initializer, gamma_initializer):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, 4, 4, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1, 1, 1])
    layer = ed.layers.Conv2DRank1(
        output_dim,
        kernel_size=2,
        padding="same",
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        activation=None,
        ensemble_size=ensemble_size)

    output = layer(batched_inputs)
    manual_output = [
        super(ed.layers.Conv2DRank1, layer).call(inputs * layer.alpha[i]) *
        layer.gamma[i] + layer.ensemble_bias[i] for i in range(ensemble_size)
    ]
    manual_output = tf.concat(manual_output, axis=0)
    self.assertEqual(output.shape,
                     (ensemble_size*examples_per_model, 4, 4, output_dim))
    self.assertAllClose(output, manual_output)

  @parameterized.parameters(
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "trainable_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "trainable_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
  )
  def testConv2DRank1AlphaGamma(self,
                                alpha_initializer,
                                gamma_initializer,
                                all_close,
                                use_additive_perturbation,
                                ensemble_size):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5*ensemble_size, 4, 4, 12).astype(np.float32)
    model = ed.layers.Conv2DRank1(
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

  @parameterized.parameters(
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal"},
      {"alpha_initializer": "trainable_deterministic",
       "gamma_initializer": "trainable_deterministic"},
  )
  def testConv1DRank1BatchEnsemble(self, alpha_initializer, gamma_initializer):
    tf.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, 4, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1, 1])
    layer = ed.layers.Conv1DRank1(
        output_dim,
        kernel_size=2,
        padding="same",
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        activation=None,
        ensemble_size=ensemble_size)

    output = layer(batched_inputs)
    manual_output = [
        super(ed.layers.Conv1DRank1, layer).call(inputs * layer.alpha[i]) *
        layer.gamma[i] + layer.ensemble_bias[i] for i in range(ensemble_size)
    ]
    manual_output = tf.concat(manual_output, axis=0)
    self.assertEqual(output.shape,
                     (ensemble_size*examples_per_model, 4, output_dim))
    self.assertAllClose(output, manual_output)

  @parameterized.parameters(
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "zeros",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "trainable_normal",
       "all_close": True,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "zeros",
       "gamma_initializer": "trainable_normal",
       "all_close": True,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": False,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": True,
       "ensemble_size": 1},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": False,
       "ensemble_size": 4},
      {"alpha_initializer": "trainable_normal",
       "gamma_initializer": "trainable_normal",
       "all_close": False,
       "use_additive_perturbation": True,
       "ensemble_size": 4},
  )
  def testConv1DRank1AlphaGamma(self,
                                alpha_initializer,
                                gamma_initializer,
                                all_close,
                                use_additive_perturbation,
                                ensemble_size):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5*ensemble_size, 4, 12).astype(np.float32)
    model = ed.layers.Conv1DRank1(
        4,
        kernel_size=2,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        activation=None)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.assertEqual(outputs1.shape, (5*ensemble_size, 3, 4))
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()


if __name__ == "__main__":
  tf.test.main()
