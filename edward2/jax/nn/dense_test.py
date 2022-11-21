# coding=utf-8
# Copyright 2022 The Edward2 Authors.
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

"""Tests for jax_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import edward2.jax as ed
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


class DenseTest(parameterized.TestCase):

  @parameterized.parameters(
      (4, [3, 20], 0.0, 10),     # 2-Dimensional input
      (2, [5, 7, 3], -0.5, 10),  # 3-Dimensional input
      (2, [5, 7, 2, 3], -0.5, 10),  # 4-Dimensional input
  )
  def test_dense(self, ens_size, inputs_shape, random_sign_init, output_dim):
    alpha_init = ed.nn.utils.make_sign_initializer(random_sign_init)
    gamma_init = ed.nn.utils.make_sign_initializer(random_sign_init)
    inputs = jax.random.normal(jax.random.PRNGKey(0), inputs_shape,
                               dtype=jnp.float32)
    tiled_inputs = jnp.tile(inputs, [ens_size] + [1] * (inputs.ndim - 1))

    layer = ed.nn.DenseBatchEnsemble(
        features=output_dim,
        ens_size=ens_size,
        alpha_init=alpha_init,
        gamma_init=gamma_init)

    tiled_outputs, params = layer.init_with_output(jax.random.PRNGKey(0),
                                                   tiled_inputs)
    params_shape = jax.tree_map(lambda x: x.shape, params)
    expected_kernel_shape = (inputs_shape[-1], output_dim)
    expected_alpha_shape = (ens_size, inputs_shape[-1])
    expected_gamma_shape = (ens_size, output_dim)
    self.assertEqual(expected_kernel_shape, params_shape["params"]["kernel"])
    self.assertEqual(expected_alpha_shape,
                     params_shape["params"]["fast_weight_alpha"])
    self.assertEqual(expected_gamma_shape,
                     params_shape["params"]["fast_weight_gamma"])

    loop_outputs = []
    for i in range(ens_size):
      alpha_shape = (1,) * (inputs.ndim - 1) + (-1,)
      alpha = params["params"]["fast_weight_alpha"][i].reshape(alpha_shape)
      scaled_inputs = inputs * alpha
      outputs = jnp.dot(scaled_inputs, params["params"]["kernel"])
      loop_outputs.append(outputs * params["params"]["fast_weight_gamma"][i] +
                          params["params"]["bias"][i])
    loop_outputs_list = jnp.concatenate(loop_outputs, axis=0)
    expected_outputs_shape = tiled_inputs.shape[:-1] + (output_dim,)
    self.assertEqual(tiled_outputs.shape, expected_outputs_shape)
    np.testing.assert_allclose(tiled_outputs, loop_outputs_list,
                               rtol=1e-06, atol=1e-06)

  @parameterized.parameters(
      (4, [3, 20], 0.0, 10),     # 2-Dimensional input
      (2, [5, 7, 3], -0.5, 10),  # 3-Dimensional input
      (2, [5, 7, 2, 3], -0.5, 10),  # 4-Dimensional input
      (2, [5, 7, 3], -0.5, (5, 2)),  # 2-Dimensional features
      (2, [5, 7, 3], -0.5, (5, 2, 3)),  # 3-Dimensional features
  )
  def test_dense_general(self, ens_size, inputs_shape, random_sign_init,
                         features):
    """Tests with variable-shape features, int axis, and no batch dims."""
    if isinstance(features, int):
      features = (features,)
    alpha_init = ed.nn.utils.make_sign_initializer(random_sign_init)
    gamma_init = ed.nn.utils.make_sign_initializer(random_sign_init)
    inputs = jax.random.normal(jax.random.PRNGKey(0), inputs_shape,
                               dtype=jnp.float32)
    # Expand input with an ensemble size dimension (it will be broadcasted).
    tiled_inputs = jnp.expand_dims(inputs, axis=0)

    layer = ed.nn.DenseGeneralBatchEnsemble(
        features=features,
        axis=-1,
        ens_size=ens_size,
        alpha_init=alpha_init,
        gamma_init=gamma_init)

    tiled_outputs, params = layer.init_with_output(jax.random.PRNGKey(0),
                                                   tiled_inputs)
    params_shape = jax.tree_map(lambda x: x.shape, params)
    expected_kernel_shape = (inputs_shape[-1],) + features
    expected_alpha_shape = (ens_size, inputs_shape[-1])
    expected_gamma_shape = (ens_size,) + features
    expected_bias_shape = (ens_size,) + features
    self.assertEqual(expected_kernel_shape, params_shape["params"]["kernel"])
    self.assertEqual(expected_alpha_shape,
                     params_shape["params"]["fast_weight_alpha"])
    self.assertEqual(expected_gamma_shape,
                     params_shape["params"]["fast_weight_gamma"])
    self.assertEqual(expected_bias_shape,
                     params_shape["params"]["bias"])

    loop_outputs = []
    for i in range(ens_size):
      alpha_shape = (1,) * (inputs.ndim - 1) + (-1,)
      alpha = params["params"]["fast_weight_alpha"][i].reshape(alpha_shape)
      scaled_inputs = inputs * alpha
      outputs = lax.dot_general(scaled_inputs, params["params"]["kernel"],
                                (((-1 + inputs.ndim,), (0,)), ((), ())))
      gamma_shape = (1,) * (outputs.ndim - len(features)) + features
      gamma = params["params"]["fast_weight_gamma"][i].reshape(gamma_shape)
      loop_outputs.append(outputs * gamma + params["params"]["bias"][i])
    loop_outputs_list = jnp.stack(loop_outputs, axis=0)
    expected_outputs_shape = (ens_size,) + tiled_inputs.shape[1:-1] + features
    self.assertEqual(tiled_outputs.shape, expected_outputs_shape)
    np.testing.assert_allclose(tiled_outputs, loop_outputs_list,
                               rtol=1e-06, atol=1e-06)


if __name__ == "__main__":
  absltest.main()
