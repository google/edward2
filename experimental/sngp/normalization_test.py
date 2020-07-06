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
"""Tests for normalization functions.

## References:

[1] Hanie Sedghi, Vineet Gupta, Philip M. Long.
    The Singular Values of Convolutional Layers.
    In _International Conference on Learning Representations_, 2019.
"""

from absl.testing import parameterized
from experimental.sngp import normalization as nm  # local file import

import numpy as np
import tensorflow as tf

DenseLayer = tf.keras.layers.Dense(10)
Conv2DLayer = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='SAME')


def _compute_spectral_norm(weight):
  if weight.ndim > 2:
    # Computes Conv2D via FFT transform as in [1].
    weight = np.fft.fft2(weight, weight.shape[1:3], axes=[0, 1])
  return np.max(np.linalg.svd(weight, compute_uv=False))


class NormalizationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(NormalizationTest, self).setUp()
    self.num_iterations = 1000
    self.norm_multiplier = 0.95

  @parameterized.named_parameters(
      ('Dense', (16, 10), DenseLayer, nm.SpectralNormalization),
      ('Conv2D', (16, 32, 32, 3), Conv2DLayer, nm.SpectralNormalizationConv2D))
  def test_spec_norm_magnitude(self, input_shape, layer, norm_wrapper):
    """Tests if the weights spectral norm converges to norm_multiplier."""
    layer.build(input_shape)
    sn_layer = norm_wrapper(layer,
                            iteration=self.num_iterations,
                            norm_multiplier=self.norm_multiplier)

    # Perform normalization.
    sn_layer.build(input_shape)
    sn_layer.update_weights()
    normalized_kernel = sn_layer.layer.kernel.numpy()

    spectral_norm_computed = _compute_spectral_norm(normalized_kernel)
    spectral_norm_expected = self.norm_multiplier
    self.assertAllClose(
        spectral_norm_computed, spectral_norm_expected, atol=5e-2)


if __name__ == '__main__':
  tf.test.main()
