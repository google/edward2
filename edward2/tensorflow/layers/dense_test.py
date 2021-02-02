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

"""Tests for Bayesian dense layers."""

import itertools
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class DenseTest(parameterized.TestCase, tf.test.TestCase):

  def testTrainableNormalStddevConstraint(self):
    layer = ed.layers.DenseReparameterization(
        100, kernel_initializer="trainable_normal")
    inputs = tf.random.normal([1, 1])
    _ = layer(inputs)
    stddev = layer.kernel.distribution.stddev()
    self.assertAllGreater(stddev, 0.)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  def testDenseKernel(self,
                      layer,
                      kernel_initializer,
                      bias_initializer,
                      all_close):
    tf.python.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5, 3, 12).astype(np.float32)
    model = layer(4,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation=tf.nn.relu)
    outputs1 = tf.convert_to_tensor(model(inputs))
    outputs2 = tf.convert_to_tensor(model(inputs))
    self.assertEqual(outputs1.shape, (5, 3, 4))
    if layer != ed.layers.DenseDVI:
      self.assertAllGreaterEqual(outputs1, 0.)
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    model.get_config()

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
  )
  def testDenseMean(self, layer):
    """Tests that forward pass can use other values, e.g., posterior mean."""
    tf.python.keras.backend.set_learning_phase(0)  # test time
    def take_mean(f, *args, **kwargs):
      """Sets random variable value to its mean."""
      rv = f(*args, **kwargs)
      rv._value = rv.distribution.mean()
      return rv
    inputs = np.random.rand(5, 3, 7).astype(np.float32)
    model = layer(4, activation=tf.nn.relu, use_bias=False)
    outputs1 = tf.convert_to_tensor(model(inputs))
    with ed.trace(take_mean):
      outputs2 = tf.convert_to_tensor(model(inputs))
    self.assertEqual(outputs1.shape, (5, 3, 4))
    self.assertNotAllClose(outputs1, outputs2)
    if layer != ed.layers.DenseDVI:
      self.assertAllClose(outputs2, np.zeros((5, 3, 4)), atol=1e-4)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
  )
  def testDenseLoss(self, layer):
    tf.python.keras.backend.set_learning_phase(1)  # training time
    features = np.random.rand(5, 12).astype(np.float32)
    labels = np.random.rand(5, 10).astype(np.float32)
    model = layer(10)

    # Imagine this is the 1st epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions = model(features)  # first call forces build
      model(features)  # ensure robustness after multiple calls
      nll = tf.python.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      # Note in TF 2.0, checking membership (v in model.weights) raises an error
      # for lists of differently shaped Tensors.
      self.assertTrue(any(v is weight for weight in model.weights))

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
      predictions = model(features)  # build is not called
      nll = tf.python.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      # Note in TF 2.0, checking membership (v in model.weights) raises an error
      # for lists of differently shaped Tensors.
      self.assertTrue(any(v is weight for weight in model.weights))

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
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
      {"layer": ed.layers.DenseRank1},
  )
  def testDenseModel(self, layer):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.python.keras.Sequential([
        tf.python.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.python.keras.layers.Flatten(),
        layer(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    if layer == ed.layers.DenseHierarchical:
      self.assertLen(model.losses, 3)
    elif layer == ed.layers.DenseRank1:
      self.assertLen(model.losses, 2)
    else:
      self.assertLen(model.losses, 1)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
  )
  def testDenseSubclass(self, layer):
    class DenseSubclass(layer):
      pass

    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.python.keras.Sequential([
        tf.python.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.python.keras.layers.Flatten(),
        DenseSubclass(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.assertEqual(outputs.shape, (3, 2))
    if layer == ed.layers.DenseHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  def testDenseDVIIsDeterministic(self):
    """Tests that DenseDVI network has a deterministic loss function."""
    features = np.random.rand(3, 2).astype(np.float32)
    labels = np.random.rand(3, 1).astype(np.float32)
    model = tf.python.keras.Sequential([
        ed.layers.DenseDVI(5, activation=tf.nn.relu),
        ed.layers.DenseDVI(1, activation=None),
    ])
    def loss_fn(features, labels):
      outputs = model(features, training=True)
      nll = -tf.reduce_sum(outputs.distribution.log_prob(labels))
      kl = sum(model.losses)
      return nll + kl
    self.assertEqual(loss_fn(features, labels), loss_fn(features, labels))

  def testDenseDVIMoments(self):
    """Verifies DenseDVI's moments empirically with samples."""
    tf.random.set_seed(377269)
    batch_size = 3
    num_features = 5
    units = 128
    num_samples = 50000
    inputs = tf.cast(np.random.rand(batch_size, num_features), dtype=tf.float32)
    layer = ed.layers.DenseDVI(units, activation=tf.nn.relu)

    outputs1 = layer(inputs)
    mean1 = outputs1.distribution.mean()
    covariance1 = outputs1.distribution.covariance()

    kernel_samples = layer.kernel.distribution.sample(num_samples)
    outputs2 = layer.activation(
        tf.einsum("bd,sdu->sbu", inputs, kernel_samples) +
        tf.reshape(layer.bias, [1, 1, units]))
    mean2 = tf.reduce_mean(outputs2, axis=0)
    centered_outputs2 = tf.transpose(a=outputs2 - mean2, perm=[1, 2, 0])
    covariance2 = tf.matmul(centered_outputs2,
                            centered_outputs2,
                            transpose_b=True) / float(num_samples)

    # Check % of mismatches is not too high according to heuristic thresholds.
    num_mismatches = np.sum(np.abs(mean1 - mean2) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units)
    self.assertLessEqual(percent_mismatches, 0.05)
    num_mismatches = np.sum(np.abs(covariance1 - covariance2) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units * units)
    self.assertLessEqual(percent_mismatches, 0.05)

  def testDenseBatchEnsemble(self):
    """Tests that vectorized implementation is same as for loop."""
    tf.python.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, input_dim])
    layer = ed.layers.DenseBatchEnsemble(
        output_dim,
        alpha_initializer="he_normal",
        gamma_initializer="he_normal",
        activation=None,
        ensemble_size=ensemble_size)
    batch_inputs = tf.tile(inputs, [ensemble_size, 1])
    batch_outputs = layer(batch_inputs)

    loop_outputs = []
    for i in range(ensemble_size):
      outputs = super(ed.layers.DenseBatchEnsemble, layer).call(
          inputs * layer.alpha[i])
      loop_outputs.append(outputs * layer.gamma[i] + layer.ensemble_bias[i])

    loop_outputs_list = tf.concat(loop_outputs, axis=0)

    expected_shape = (ensemble_size * examples_per_model, output_dim)
    self.assertEqual(batch_outputs.shape, expected_shape)
    self.assertAllClose(batch_outputs, loop_outputs_list)

  @parameterized.parameters(
      itertools.product([True, False], [True, False], [True, False]))
  def testDenseHyperBatchEnsemble(self, use_bias, regularize_fast_weights,
                                  fast_weights_eq_constraint):
    tf.random.set_seed(1)

    units = 5
    lambda_key_to_index = {"self_dense_l2_kernel": 0, "self_dense_l2_bias": 1}
    ens_size = 3

    layer = ed.layers.DenseHyperBatchEnsemble(
        units,
        lambda_key_to_index,
        ensemble_size=ens_size,
        name="self_dense",
        activation=None,
        use_bias=use_bias,
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",
        alpha_initializer="glorot_uniform",
        gamma_initializer="glorot_uniform",
        regularize_fast_weights=regularize_fast_weights,
        fast_weights_eq_contraint=fast_weights_eq_constraint)

    n = 6
    e = tf.random.normal((n * ens_size, 2*units))
    lambdas = tf.random.uniform((n * ens_size, 2))

    dim_x = 4
    x = tf.random.normal((n, dim_x))
    tile_x = tf.tile(x, [ens_size, 1])

    outputs = layer([tile_x, lambdas, e])

    dense_kernel = layer.dense.kernel
    delta_dense_kernel = layer.delta_dense.kernel

    expected_outputs = []
    for k, ek in zip(range(ens_size), tf.split(e, ens_size)):

      r_k = tf.reshape(layer.dense.alpha[k, :], (dim_x, 1))
      s_k = tf.reshape(layer.dense.gamma[k, :], (1, units))

      u_k = tf.reshape(layer.delta_dense.alpha[k, :], (dim_x, 1))
      v_k = tf.reshape(layer.delta_dense.gamma[k, :], (1, units))

      if fast_weights_eq_constraint:
        self.assertAllClose(r_k, u_k)
        self.assertAllClose(s_k, v_k)

      for x_i, e_i in zip(x, ek):
        x_i = tf.reshape(x_i, (1, dim_x))
        e1_i = tf.reshape(e_i[:units], (1, units))
        e2_i = tf.reshape(e_i[units:], (units,))

        kernel_i = dense_kernel * r_k * s_k
        delta_kernel_i = (delta_dense_kernel * u_k * v_k) * e1_i

        expected_outputs_i = tf.matmul(x_i, kernel_i + delta_kernel_i)

        if use_bias:
          bias_i = layer.dense.ensemble_bias[k, :]
          delta_bias_i = layer.bias[k, :] * e2_i
          expected_outputs_i += bias_i + delta_bias_i

        expected_outputs.append(expected_outputs_i)

    self.assertAllClose(outputs, tf.concat(expected_outputs, 0))

    mean_l2_regularizer = 0.
    for k, ek, lambdask in zip(
        range(ens_size), tf.split(e, ens_size), tf.split(lambdas, ens_size)):

      r_k = tf.reshape(layer.dense.alpha[k, :], (dim_x, 1))
      s_k = tf.reshape(layer.dense.gamma[k, :], (1, units))

      u_k = tf.reshape(layer.delta_dense.alpha[k, :], (dim_x, 1))
      v_k = tf.reshape(layer.delta_dense.gamma[k, :], (1, units))

      for lambdas_i, e_i in zip(lambdask, ek):
        l2_kernel, l2_bias = lambdas_i[0], lambdas_i[1]
        e1_i = tf.reshape(e_i[:units], (1, units))
        e2_i = tf.reshape(e_i[units:], (units,))

        if regularize_fast_weights:
          kernel_i = dense_kernel * r_k * s_k
          delta_kernel_i = (delta_dense_kernel * u_k * v_k) * e1_i
        else:
          kernel_i = dense_kernel
          delta_kernel_i = delta_dense_kernel * e1_i

        mean_l2_regularizer += l2_kernel * tf.reduce_sum(
            tf.square(kernel_i + delta_kernel_i))

        if use_bias:
          bias_i = layer.dense.ensemble_bias[k, :]
          delta_bias_i = layer.bias[k, :] * e2_i
          mean_l2_regularizer += l2_bias * tf.reduce_sum(
              tf.square(bias_i + delta_bias_i))

    mean_l2_regularizer *= 1. / (n * ens_size)

    self.assertAllClose(float(mean_l2_regularizer), float(layer.losses[0]))

  @parameterized.parameters(
      {"alpha_initializer": "he_normal",
       "gamma_initializer": "he_normal",
       "bias_initializer": "zeros"},
      {"alpha_initializer": "trainable_deterministic",
       "gamma_initializer": "trainable_deterministic",
       "bias_initializer": "trainable_deterministic"},
  )
  def testDenseRank1BatchEnsemble(self,
                                  alpha_initializer,
                                  gamma_initializer,
                                  bias_initializer):
    tf.python.keras.backend.set_learning_phase(1)  # training time
    ensemble_size = 3
    examples_per_model = 4
    input_dim = 5
    output_dim = 5
    inputs = tf.random.normal([examples_per_model, input_dim])
    batched_inputs = tf.tile(inputs, [ensemble_size, 1])
    layer = ed.layers.DenseRank1(
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
        super(ed.layers.DenseRank1, layer).call(inputs * layer.alpha[i]) *
        layer.gamma[i] + layer.ensemble_bias[i] for i in range(ensemble_size)
    ]
    manual_output = tf.concat(manual_output, axis=0)

    expected_shape = (ensemble_size*examples_per_model, output_dim)
    self.assertEqual(output.shape, expected_shape)
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
  def testDenseRank1AlphaGamma(self,
                               alpha_initializer,
                               gamma_initializer,
                               all_close,
                               use_additive_perturbation,
                               ensemble_size):
    tf.python.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5*ensemble_size, 12).astype(np.float32)
    model = ed.layers.DenseRank1(
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

  def testCondConv(self):
    tf.python.keras.backend.set_learning_phase(1)  # training time
    features = np.random.rand(5, 12).astype(np.float32)
    routing_weights = np.random.rand(5, 3).astype(np.float32)
    model = ed.layers.CondDense(10, num_experts=3)
    predictions = model(features, routing_weights)
    self.assertEqual(predictions.shape, (5, 10))
    conddense_kernel = model.trainable_weights[0]
    expert_kernels = tf.reshape(conddense_kernel, [3, 12, 10])
    expert_outputs = []
    for idx in range(3):
      expert_outputs.append(tf.linalg.matmul(features, expert_kernels[idx]))
    expert_outputs = tf.stack(expert_outputs, axis=1)
    routing_weights_3d = tf.expand_dims(routing_weights, 1)  # [5, 1, 3]
    manual_predictions = tf.linalg.matmul(routing_weights_3d, expert_outputs)
    manual_predictions = tf.squeeze(manual_predictions)  # [5, 10]
    self.assertEqual(predictions.shape, manual_predictions.shape)
    self.assertAllClose(predictions, manual_predictions)


if __name__ == "__main__":
  tf.test.main()
