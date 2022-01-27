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

"""Tests for random_feature."""
import functools

from absl.testing import absltest
from absl.testing import parameterized

import edward2.jax as ed

import flax.linen as nn

import jax
import jax.numpy as jnp
import numpy as np


def exp_quadratic(x1, x2):
  return jnp.exp(-jnp.sum((x1 - x2)**2) / 2.)


def linear(x1, x2):
  return jnp.sum(x1 * x2)


def cov_map(xs, xs2=None, cov_func=None):
  """Compute a covariance matrix from a covariance function and data points.

  Args:
    xs: array of data points, stacked along the leading dimension.
    xs2: second array of data points, stacked along the non-leading dimension.
    cov_func: callable function, maps pairs of data points to scalars.

  Returns:
    A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
  """
  if xs2 is None:
    return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs)
  else:
    return jax.vmap(lambda x: jax.vmap(lambda y: cov_func(x, y))(xs))(xs2).T


def _compute_posterior_kernel(x_tr, x_ts, ridge_penalty, kernel_func=None):
  """Computes the posterior covariance matrix of a Gaussian process."""
  if kernel_func is None:
    kernel_func = functools.partial(cov_map, cov_func=linear)

  num_sample = x_tr.shape[0]

  k_tt = kernel_func(x_tr)
  k_tt_ridge = k_tt + ridge_penalty * jnp.eye(num_sample)

  k_ts = kernel_func(x_tr, x_ts)
  k_tt_inv_k_ts = jnp.linalg.solve(k_tt_ridge, k_ts)

  k_ss = kernel_func(x_ts)

  return k_ss - jnp.matmul(jnp.transpose(k_ts), k_tt_inv_k_ts)


def _generate_normal_data(num_sample, num_dim, loc=0., seed=None):
  """Generates random data sampled from i.i.d. normal distribution."""
  np.random.seed(seed)
  return np.random.normal(
      size=(num_sample, num_dim), loc=loc, scale=1. / np.sqrt(num_dim))


class RandomFeatureGaussianProcessTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = 0

    self.ridge_penalty = 1.
    self.num_data_dim = 4
    self.num_test_sample = 16
    self.num_train_sample = 2000
    self.num_random_features = 1024
    self.rbf_func = functools.partial(cov_map, cov_func=exp_quadratic)

    self.x_train = _generate_normal_data(
        self.num_train_sample, self.num_data_dim, seed=12)
    self.x_test = _generate_normal_data(
        self.num_test_sample, self.num_data_dim, seed=21)

    # Uses classic RBF random feature distribution.
    self.hidden_kwargs = dict(
        kernel_init=nn.initializers.normal(stddev=1.), feature_scale=None)

    self.rbf_approx_maximum_tol = 5e-3
    self.rbf_approx_average_tol = 5e-4
    self.primal_dual_maximum_diff = 1e-6
    self.primal_dual_average_diff = 1e-7

  def one_step_rfgp_result(self, train_data, test_data, **eval_kwargs):
    """Returns the RFGP result after one-step covariance update."""
    rfgp = ed.nn.RandomFeatureGaussianProcess(
        features=1,
        hidden_features=self.num_random_features,
        normalize_input=False,
        hidden_kwargs=self.hidden_kwargs,
        covmat_kwargs=dict(ridge_penalty=self.ridge_penalty))

    # Computes posterior covariance on test data.
    init_key = jax.random.PRNGKey(self.seed)
    init_variables = rfgp.init(init_key, inputs=train_data)
    state, params = init_variables.pop('params')
    del init_variables

    # Perform one-step update on training data.
    unused_rfgp_logits_train, updated_state = rfgp.apply(
        {
            'params': params,
            **state
        },
        inputs=train_data,
        mutable=list(state.keys()))
    del unused_rfgp_logits_train

    # Returns the evaluate result on test data.
    # Note we don't specify mutable collection during eval.
    updated_variables = {'params': params, **updated_state}
    return rfgp.apply(updated_variables, inputs=test_data, **eval_kwargs)

  def test_rfgp_posterior_approximation_exact_rbf(self):
    """Tests if posterior covmat approximates that from a RBF model."""
    # Evaluates on test data.
    _, rfgp_covmat_test = self.one_step_rfgp_result(
        self.x_train, self.x_test, return_full_covmat=True)

    # Compares with exact RBF posterior covariance.
    rbf_covmat_test = _compute_posterior_kernel(self.x_train, self.x_test,
                                                self.ridge_penalty,
                                                self.rbf_func)
    covmat_maximum_diff = jnp.max(jnp.abs(rbf_covmat_test - rfgp_covmat_test))
    covmat_average_diff = jnp.mean(jnp.abs(rbf_covmat_test - rfgp_covmat_test))

    self.assertLess(covmat_maximum_diff, self.rbf_approx_maximum_tol)
    self.assertLess(covmat_average_diff, self.rbf_approx_average_tol)

  def test_rfgp_posterior_approximation_dual_form(self):
    """Tests if the primal-form posterior matches with the dual form."""
    # Computes the covariance matrix using primal-form formula.
    x_train = _generate_normal_data(128, self.num_data_dim)
    x_test = _generate_normal_data(64, self.num_data_dim)

    _, _, rfgp_features_train = self.one_step_rfgp_result(
        train_data=x_train, test_data=x_train,
        return_full_covmat=True, return_random_features=True)
    _, rfgp_covmat_primal, rfgp_features_test = self.one_step_rfgp_result(
        train_data=x_train, test_data=x_test,
        return_full_covmat=True, return_random_features=True)

    # Computing random feature posterior covariance using primal formula.
    linear_kernel_func = functools.partial(cov_map, cov_func=linear)
    rfgp_covmat_dual = _compute_posterior_kernel(
        rfgp_features_train, rfgp_features_test,
        ridge_penalty=self.ridge_penalty,
        kernel_func=linear_kernel_func)

    covmat_diff = jnp.abs(rfgp_covmat_dual - rfgp_covmat_primal)
    covmat_maximum_diff = jnp.max(covmat_diff)
    covmat_average_diff = jnp.mean(covmat_diff)

    self.assertLess(covmat_maximum_diff, self.primal_dual_maximum_diff)
    self.assertLess(covmat_average_diff, self.primal_dual_average_diff)

  @parameterized.named_parameters(
      ('diag_covmat_no_rff', False, False),
      ('diag_covmat_with_rff', False, True),
      ('full_covmat_no_rff', True, False),
      ('full_covmat_with_rff', True, True),
  )
  def test_rfgp_output_shape(self, return_full_covmat, return_random_features):
    """Tests if the shape of output covmat and random features are correct."""
    rfgp_results = self.one_step_rfgp_result(
        train_data=self.x_train,
        test_data=self.x_test,
        return_full_covmat=return_full_covmat,
        return_random_features=return_random_features)

    expected_results_len = 2 + return_random_features
    observed_covmat_shape = rfgp_results[1].shape
    expected_covmat_shape = ((self.num_test_sample,) if not return_full_covmat
                             else (self.num_test_sample, self.num_test_sample))
    self.assertLen(rfgp_results, expected_results_len)
    self.assertEqual(observed_covmat_shape, expected_covmat_shape)

    if return_random_features:
      expected_feature_shape = (self.num_test_sample, self.num_random_features)
      observed_feature_shape = rfgp_results[2].shape
      self.assertEqual(expected_feature_shape, observed_feature_shape)

  def test_rfgp_default_parameter_collections(self):
    rfgp = ed.nn.RandomFeatureGaussianProcess(
        features=1, hidden_features=self.num_random_features)

    # Computes posterior covariance on test data.
    init_key = jax.random.PRNGKey(self.seed)
    init_variables = rfgp.init(init_key, inputs=self.x_train)
    state, params = init_variables.pop('params')
    del init_variables

    # Note: the norm_layer should not show up in `param`
    # since by default it does not have trainable parameters.
    self.assertEqual(list(params.keys()), ['output_layer'])
    self.assertEqual(
        list(state.keys()), ['random_features', 'laplace_covariance'])


class RandomFeatureTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.seed = 0
    self.collection_name = 'random_fourier_features'

    self.num_data_dim = 128
    self.num_train_sample = 512
    self.num_random_features = 10240
    self.rbf_kern_func = functools.partial(cov_map, cov_func=exp_quadratic)

    self.x_train = _generate_normal_data(self.num_train_sample,
                                         self.num_data_dim)
    self.x_test = _generate_normal_data(self.num_train_sample,
                                        self.num_data_dim)

    # Uses classic RBF random feature distribution.
    self.hidden_kwargs = dict(
        kernel_init=nn.initializers.normal(stddev=1.), feature_scale=None)

    self.kernel_approx_tolerance = dict(atol=5e-2, rtol=1e-2)

  def test_random_feature_mutable_collection(self):
    """Tests if RFF variables are properly nested under a mutable collection."""
    rng = jax.random.PRNGKey(self.seed)
    rff_layer = ed.nn.RandomFourierFeatures(
        features=self.num_random_features,
        collection_name=self.collection_name,
        **self.hidden_kwargs)

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
    rff_layer = ed.nn.RandomFourierFeatures(
        features=self.num_random_features, **self.hidden_kwargs)
    y, _ = rff_layer.init_with_output(rng, x)

    expected_output_shape = input_shape[:-1] + (self.num_random_features,)
    self.assertEqual(y.shape, expected_output_shape)

  def test_random_feature_kernel_approximation(self):
    """Tests if default RFF layer approximates a RBF kernel matrix."""
    rng = jax.random.PRNGKey(self.seed)
    rff_layer = ed.nn.RandomFourierFeatures(
        features=self.num_random_features,
        collection_name=self.collection_name,
        **self.hidden_kwargs)

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
