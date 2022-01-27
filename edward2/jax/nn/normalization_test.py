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

"""Tests for normalization layers, ported from the tensorflow implementation.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""
from absl.testing import absltest
from absl.testing import parameterized

import edward2.jax as ed
import flax
import flax.linen as nn
import jax
import numpy as np

DenseLayer = nn.Dense(10)
Conv2DLayer = nn.Conv(features=64, kernel_size=(3, 3), padding="VALID")


def _compute_spectral_norm(weight, input_shape):
  if weight.ndim > 2:
    # Computes Conv2D via FFT transform as in [1].
    weight = np.fft.fft2(weight, input_shape[1:3], axes=[0, 1])
  return np.max(np.linalg.svd(weight, compute_uv=False))


class NormalizationTest(parameterized.TestCase):

  def setUp(self):
    super(NormalizationTest, self).setUp()
    self.num_iterations = 1000
    self.norm_multiplier = 0.95

  @parameterized.named_parameters(
      ("Dense", (None, 10), DenseLayer, ed.nn.SpectralNormalization),
      ("Conv2D",
       (None, 32, 32, 3), Conv2DLayer, ed.nn.SpectralNormalizationConv2D))
  def test_spec_norm_magnitude(self, input_shape, layer, norm_wrapper):
    """Tests if the weights spectral norm converges to norm_multiplier."""
    sn_layer = norm_wrapper(
        layer,
        iteration=self.num_iterations,
        norm_multiplier=self.norm_multiplier)

    # Perform normalization.
    inputs = np.random.uniform(size=input_shape[1:])
    init_state = sn_layer.init(jax.random.PRNGKey(1), inputs)
    kernel = init_state["params"][type(layer).__name__]["kernel"]
    _, state = sn_layer.apply(
        init_state, inputs, mutable=["spectral_stats", "intermediates"])
    normalized_kernel = state["intermediates"]["w"][0]

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel,
                                                    input_shape)
    spectral_norm_original = _compute_spectral_norm(kernel, input_shape)
    spectral_norm_expected = min(spectral_norm_original, self.norm_multiplier)
    np.testing.assert_allclose(
        spectral_norm_computed, spectral_norm_expected, atol=1e-2)

    # We will scale the kernel so that its spectral norm is equal to a target
    # value greater than 1. and check if the computed sigma is near that value.
    spectral_norm_target = 10.
    new_kernel = kernel * (spectral_norm_target / spectral_norm_original)
    new_init_state = flax.core.unfreeze(init_state)
    new_init_state["params"][type(layer).__name__]["kernel"] = new_kernel
    new_init_state = flax.core.freeze(new_init_state)
    _, new_state = sn_layer.apply(
        new_init_state, inputs, mutable=["spectral_stats", "intermediates"])
    new_normalized_kernel = new_state["intermediates"]["w"][0]
    sigma_computed = (new_kernel /
                      new_normalized_kernel).mean() * self.norm_multiplier
    np.testing.assert_allclose(sigma_computed, spectral_norm_target, rtol=1e-2)

    # Test that the normalized layer is K-Lipschitz. In particular, if the layer
    # is a function f, then ||f(x1) - f(x2)||_2 <= K * ||(x1 - x2)||_2, where K
    # is the norm multiplier.
    new_input_shape = (16,) + input_shape[1:]
    new_input = np.random.uniform(size=new_input_shape)
    delta_vec = np.random.uniform(size=new_input_shape)
    output1, _ = sn_layer.apply(init_state, new_input, mutable="spectral_stats")
    output2, _ = sn_layer.apply(
        init_state, new_input + delta_vec, mutable="spectral_stats")

    delta_input = np.linalg.norm(delta_vec)
    delta_output = np.linalg.norm(output2 - output1)
    self.assertLessEqual(delta_output, self.norm_multiplier * delta_input)


if __name__ == "__main__":
  absltest.main()
