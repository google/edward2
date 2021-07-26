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

"""Tests for random_feature."""

from absl.testing import absltest
from absl.testing import parameterized

import edward2 as ed_tf
import edward2.jax as ed

import jax
import jax.numpy as jnp
import numpy as np

RBF_KERN_FUNC = ed_tf.layers.gaussian_process.ExponentiatedQuadratic(
    variance=1., lengthscale=1.)


def _generate_normal_data(num_sample, num_dim, loc=0.):
  """Generates random data sampled from i.i.d. normal distribution."""
  return np.random.normal(
      size=(num_sample, num_dim), loc=loc, scale=1. / np.sqrt(num_dim))


class RandomFeatureTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = 0
    self.collection_name = 'random_fourier_features'

    self.num_data_dim = 128
    self.num_train_sample = 512
    self.num_random_features = 10240
    self.rbf_kern_func = RBF_KERN_FUNC

    self.x_train = _generate_normal_data(self.num_train_sample,
                                         self.num_data_dim)
    self.x_test = _generate_normal_data(self.num_train_sample,
                                        self.num_data_dim)

    self.kernel_approx_tolerance = dict(atol=5e-2, rtol=1e-2)

  def test_random_feature_mutable_collection(self):
    """Tests if RFF variables are properly nested under a mutable collection."""
    rng = jax.random.PRNGKey(self.seed)
    rff_layer = ed.nn.RandomFourierFeatures(
        features=self.num_random_features, collection_name=self.collection_name)

    # Computes forward pass with mutable collection specified.
    init_vars = rff_layer.init(rng, self.x_train)
    _, mutable_vars = rff_layer.apply(
        init_vars, self.x_train, mutable=[self.collection_name])

    # Check if random feature variables are in the mutable variable collection.
    rff_vars = mutable_vars[self.collection_name]
    rff_kernel = rff_vars['kernel']
    rff_bias = rff_vars['bias']

    self.assertEqual(rff_kernel.shape,
                     (self.num_data_dim, self.num_random_features))
    self.assertEqual(rff_bias.shape, (self.num_random_features,))

  @parameterized.named_parameters(
      ('3d_input_tensor', (8, 12, 64)),  # 3-Dimensional input
      ('4d_input_tensor', (8, 6, 16, 32)),  # 4-Dimensional input
  )
  def test_random_feature_nd_input(self, input_shape):
    rng = jax.random.PRNGKey(self.seed)
    x = jnp.ones(input_shape)
    rff_layer = ed.nn.RandomFourierFeatures(features=self.num_random_features)
    y, _ = rff_layer.init_with_output(rng, x)

    expected_output_shape = input_shape[:-1] + (self.num_random_features,)
    self.assertEqual(y.shape, expected_output_shape)

  def test_random_feature_kernel_approximation(self):
    """Tests if default RFF layer approximates a RBF kernel matrix."""
    rng = jax.random.PRNGKey(self.seed)
    rff_layer = ed.nn.RandomFourierFeatures(
        features=self.num_random_features, collection_name=self.collection_name)

    # Extracts random features by computing forward pass.
    init_vars = rff_layer.init(rng, self.x_train)
    random_feature, _ = rff_layer.apply(
        init_vars, self.x_train, mutable=[self.collection_name])

    # Computes the approximated and the exact kernel matrices.
    prior_kernel_computed = random_feature.dot(random_feature.T)
    prior_kernel_expected = self.rbf_kern_func(self.x_train, self.x_train)

    np.testing.assert_allclose(prior_kernel_computed, prior_kernel_expected,
                               **self.kernel_approx_tolerance)


if __name__ == '__main__':
  absltest.main()
