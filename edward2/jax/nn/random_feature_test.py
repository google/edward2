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


class LaplaceRandomFeatureCovarianceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = 0
    self.collection_name = 'laplace_covariance'
    self.num_random_features = 1024
    self.batch_size = 31
    self.ridge_penalty = 0.32

    self.kernel_approx_tolerance = dict(atol=5e-2, rtol=1e-2)

  @parameterized.named_parameters(('gaussian_multi_class', 'gaussian', 42),
                                  ('binary_univariate', 'binary_logistic', 1),
                                  ('poisson_univariate', 'poisson', 1))
  def test_laplace_covariance_shape(self, likelihood, logit_dim):
    """Tests if the shape of the covariance matrix is correct."""
    rng = jax.random.PRNGKey(self.seed)
    rff_key, logit_key, init_key = jax.random.split(rng, 3)

    gp_features = jax.random.uniform(
        rff_key, (self.batch_size, self.num_random_features))
    gp_logits = jax.random.uniform(logit_key, (self.batch_size, logit_dim))

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood=likelihood,
    )

    # Intialize and apply one update.
    init_vars = cov_layer.init(init_key, gp_features, gp_logits)
    cov_null, mutated_vars = cov_layer.apply(
        init_vars, gp_features, gp_logits, mutable=[cov_layer.collection_name])

    # Evaluate covariance.
    cov_diag = cov_layer.apply(
        mutated_vars, gp_features, gp_logits, diagonal_only=True)
    cov_mat = cov_layer.apply(
        mutated_vars, gp_features, gp_logits, diagonal_only=False)

    # No covariance is returned during mutable update.
    self.assertIsNone(cov_null)

    # Shape of returned covariance depends on diagonal_only=True / False.
    self.assertEqual(cov_diag.shape, (self.batch_size,))
    self.assertEqual(cov_mat.shape, (self.batch_size, self.batch_size))

  @parameterized.named_parameters(
      ('binary_multivariate_logit', 'binary_logistic', 3),
      ('binary_no_logit', 'binary_logistic', None),
      ('poisson_multivariate_logit', 'binary_logistic', 2),
      ('poisson_no_logit', 'poisson', None))
  def test_laplace_covariance_likelhood_error(self, likelihood, logit_dim):
    """Tests if no-Gaussian model throw error for multivariate / null logits."""
    rng = jax.random.PRNGKey(self.seed)
    rff_key, logit_key, init_key = jax.random.split(rng, 3)

    gp_features = jax.random.uniform(
        rff_key, (self.batch_size, self.num_random_features))
    gp_logits = jax.random.uniform(logit_key,
                                   (self.batch_size,
                                    logit_dim)) if logit_dim else None

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood=likelihood,
    )
    init_vars = cov_layer.init(init_key, gp_features, gp_logits)

    with self.assertRaises(ValueError):
      _ = cov_layer.apply(
          init_vars,
          gp_features,
          gp_logits,
          mutable=[cov_layer.collection_name])

  def test_laplace_covariance_gaussian_update(self):
    """Tests if orthogonal data leads to an identity covariance matrix."""
    sample_size = self.num_random_features * 2

    rng = jax.random.PRNGKey(self.seed)
    rff_key, init_key = jax.random.split(rng, 2)

    # Make orthogonal data using SVD.
    gp_features = jax.random.uniform(
        rff_key, (sample_size, self.num_random_features))
    gp_features_ortho, _, _ = jnp.linalg.svd(gp_features, full_matrices=False)

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood='gaussian',
        ridge_penalty=self.ridge_penalty)

    # Intialize and apply one update.
    init_vars = cov_layer.init(init_key, gp_features_ortho)
    _, mutated_vars = cov_layer.apply(
        init_vars, gp_features_ortho, mutable=[cov_layer.collection_name])

    # Check precision matrices after update.
    # Under exact update and Gaussian likelihood, the precision matrix should be
    # (tr(U) @ U + ridge * I) which equals to (1 + ridge) * I.
    updated_mat_computed = mutated_vars[
        cov_layer.collection_name]['precision_matrix']
    updated_mat_expected = jnp.eye(
        self.num_random_features) * (1. + self.ridge_penalty)

    np.testing.assert_allclose(updated_mat_computed, updated_mat_expected,
                               rtol=1e-5, atol=1e-5)

  @parameterized.named_parameters(('gaussian_multi_class', 'gaussian', 4),
                                  ('binary_univariate', 'binary_logistic', 1),
                                  ('poisson_univariate', 'poisson', 1))
  def test_laplace_covariance_exact_update(self, likelihood, logit_dim):
    """Tests if exact update returns correct result."""
    # Perform exact update by setting momentum to `None`.
    momentum = None

    rng = jax.random.PRNGKey(self.seed)
    rff_key, logit_key, init_key = jax.random.split(rng, 3)

    gp_features = jax.random.uniform(
        rff_key, (self.batch_size, self.num_random_features))
    gp_logits = jax.random.uniform(logit_key,
                                   (self.batch_size,
                                    logit_dim)) if logit_dim else None

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood=likelihood,
        ridge_penalty=self.ridge_penalty,
        momentum=momentum)

    # Intialize and apply one update.
    init_vars = cov_layer.init(init_key, gp_features, gp_logits)
    _, mutated_vars = cov_layer.apply(
        init_vars, gp_features, gp_logits, mutable=[cov_layer.collection_name])

    # Check precision matrices at initialization and after update.
    init_mat_computed = init_vars[cov_layer.collection_name]['precision_matrix']
    init_mat_expected = jnp.eye(self.num_random_features) * self.ridge_penalty

    updated_mat_computed = mutated_vars[
        cov_layer.collection_name]['precision_matrix']
    updated_mat_expected = cov_layer.update_precision_matrix(
        gp_features, gp_logits, 0.) + init_mat_expected

    np.testing.assert_allclose(init_mat_computed, init_mat_expected)
    np.testing.assert_allclose(updated_mat_computed, updated_mat_expected)

  @parameterized.named_parameters(
      ('gaussian_multi_class_0', 'gaussian', 4, 0.),
      ('gaussian_multi_class_0.52', 'gaussian', 4, .52),
      ('gaussian_multi_class_1', 'gaussian', 4, 1.),
      ('binary_univariate_0', 'binary_logistic', 1, 0.),
      ('binary_univariate_0.18', 'binary_logistic', 1, .18),
      ('binary_univariate_1', 'binary_logistic', 1, 1.),
      ('poisson_univariate_0', 'poisson', 1, 0.),
      ('poisson_univariate_0.73', 'poisson', 1, .73),
      ('poisson_univariate_1', 'poisson', 1, 1.))
  def test_laplace_covariance_momentum_update(self, likelihood, logit_dim,
                                              momentum):
    """Tests if momentum update is correct."""
    rng = jax.random.PRNGKey(self.seed)
    rff_key, logit_key, init_key = jax.random.split(rng, 3)

    gp_features = jax.random.uniform(
        rff_key, (self.batch_size, self.num_random_features))
    gp_logits = jax.random.uniform(logit_key,
                                   (self.batch_size,
                                    logit_dim)) if logit_dim else None

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood=likelihood,
        ridge_penalty=self.ridge_penalty,
        momentum=momentum)

    # Intialize and apply one update.
    init_vars = cov_layer.init(init_key, gp_features, gp_logits)
    _, mutated_vars = cov_layer.apply(
        init_vars, gp_features, gp_logits, mutable=[cov_layer.collection_name])

    # Check precision matrices at initialization and after update.
    init_mat_computed = init_vars[cov_layer.collection_name]['precision_matrix']
    init_mat_expected = jnp.eye(self.num_random_features) * self.ridge_penalty

    updated_mat_computed = mutated_vars[
        cov_layer.collection_name]['precision_matrix']
    updated_mat_expected = cov_layer.update_precision_matrix(
        gp_features, gp_logits, 0.) + momentum * init_mat_expected

    np.testing.assert_allclose(init_mat_computed, init_mat_expected)
    np.testing.assert_allclose(updated_mat_computed, updated_mat_expected)

  @parameterized.named_parameters(('gaussian_multi_class', 'gaussian', 4),
                                  ('binary_univariate', 'binary_logistic', 1),
                                  ('poisson_univariate', 'poisson', 1))
  def test_laplace_covariance_diagonal_covariance(self, likelihood, logit_dim):
    """Tests if computed predictive variance is the diagonal of covar matrix."""
    rng = jax.random.PRNGKey(self.seed)
    rff_key, logit_key, init_key = jax.random.split(rng, 3)

    gp_features = jax.random.uniform(
        rff_key, (self.batch_size, self.num_random_features))
    gp_logits = jax.random.uniform(logit_key, (self.batch_size, logit_dim))

    cov_layer = ed.nn.LaplaceRandomFeatureCovariance(
        hidden_features=self.num_random_features,
        likelihood=likelihood,
        ridge_penalty=self.ridge_penalty)

    # Intialize and apply one update.
    init_vars = cov_layer.init(init_key, gp_features, gp_logits)
    _, mutated_vars = cov_layer.apply(
        init_vars, gp_features, gp_logits, mutable=[cov_layer.collection_name])

    # Evaluate covariance.
    cov_diag = cov_layer.apply(
        mutated_vars, gp_features, gp_logits, diagonal_only=True)
    cov_mat = cov_layer.apply(
        mutated_vars, gp_features, gp_logits, diagonal_only=False)

    np.testing.assert_allclose(
        cov_diag, jnp.diag(cov_mat), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
