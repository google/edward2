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

"""Definitions for random feature Gaussian process layer.

## References:

[1]: Liu et al. Simple and principled uncertainty estimation with deterministic
     deep learning via distance awareness. In _Neural Information Processing
     Systems_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Xu et al. Understanding and Improving Layer Normalization.  In _Neural
     Information Processing Systems_, 2019.
     https://papers.nips.cc/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
[3]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
     Machines. In _Neural Information Processing Systems_, 2007.
     https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
[4]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
"""
import dataclasses
import functools
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

from edward2.jax.nn import dense
import flax.linen as nn

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

# Jax-related data types.
PRNGKey = Any
Shape = Iterable[int]
Dtype = type(jnp.float32)
Array = jnp.ndarray
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

# Default config for random features.
default_rbf_activation = jnp.cos
default_rbf_bias_init = nn.initializers.uniform(scale=2. * jnp.pi)
# Using "he_normal" style random feature distribution. Effectively, this is
# equivalent to approximating a RBF kernel but with the input standardized by
# its dimensionality (i.e., input_scaled = input * sqrt(2. / dim_input)) and
# empirically leads to better performance for neural network inputs.
default_rbf_kernel_init = nn.initializers.variance_scaling(
    scale=2.0, mode='fan_in', distribution='normal')

# Default field value for kwargs, to be used for data class declaration.
default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)

SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')
MIN_SCALE_MONTE_CARLO = 1e-3


class RandomFeatureGaussianProcess(nn.Module):
  """A Gaussian process layer using random Fourier features [1].

  Attributes:
    features: the number of output units.
    hidden_features: the number of hidden random fourier features.
    normalize_input: whether to normalize the input using nn.LayerNorm.
    norm_kwargs: Optional keyword arguments to the input nn.LayerNorm layer.
    hidden_kwargs: Optional keyword arguments to the random feature layer.
    output_kwargs: Optional keyword arguments to the predictive logit layer.
    covmat_kwargs: Optional keyword arguments to the predictive covmat layer.
  """
  features: int
  hidden_features: int = 1024
  normalize_input: bool = True

  # Optional keyword arguments.
  norm_kwargs: Mapping[str, Any] = default_kwarg_dict()
  hidden_kwargs: Mapping[str, Any] = default_kwarg_dict()
  output_kwargs: Mapping[str, Any] = default_kwarg_dict()
  covmat_kwargs: Mapping[str, Any] = default_kwarg_dict()

  def setup(self):
    """Defines model layers."""
    # pylint:disable=invalid-name,not-a-mapping
    if self.normalize_input:
      # Prefer a parameter-free version of LayerNorm by default [2]. Can be
      # overwritten by passing norm_kwargs=dict(use_bias=..., use_scales=...).
      LayerNorm = functools.partial(
          nn.LayerNorm, use_bias=False, use_scale=False)
      self.norm_layer = LayerNorm(**self.norm_kwargs)

    self.hidden_layer = RandomFourierFeatures(
        features=self.hidden_features, **self.hidden_kwargs)

    self.output_layer = nn.Dense(features=self.features, **self.output_kwargs)
    self.covmat_layer = LaplaceRandomFeatureCovariance(
        hidden_features=self.hidden_features, **self.covmat_kwargs)
    # pylint:enable=invalid-name,not-a-mapping

  def __call__(self,
               inputs: Array,
               return_full_covmat: bool = False,
               return_random_features: bool = False) -> Array:
    """Computes Gaussian process outputs.

    Args:
      inputs: the nd-array of shape (batch_size, ..., input_dim).
      return_full_covmat: whether to return the full covariance matrix, shape
        (batch_size, batch_size), or only return the predictive variances with
        shape (batch_size, ).
      return_random_features: whether to return the random fourier features for
        the inputs.

    Returns:
      A tuple of predictive logits, predictive covmat and (optionally)
      random Fourier features.
    """
    gp_inputs = self.norm_layer(inputs) if self.normalize_input else inputs
    gp_features = self.hidden_layer(gp_inputs)

    gp_logits = self.output_layer(gp_features)
    gp_covmat = self.covmat_layer(
        gp_features, gp_logits, diagonal_only=not return_full_covmat)

    # Returns predictive logits, covmat and (optionally) random features.
    if return_random_features:
      return gp_logits, gp_covmat, gp_features
    return gp_logits, gp_covmat


class RandomFourierFeatures(nn.Module):
  """A random fourier feature (RFF) layer that approximates a kernel model.

  The random feature transformation is a one-hidden-layer network with
  non-trainable weights (see, e.g., Algorithm 1 of [3]). Specifically:

  f(x) = activation(x @ kernel + bias) * output_scale.

  The forward pass logic closely follows that of the nn.Dense.

  Attributes:
    features: the number of output units.
    feature_scalefeature_scale: scale to apply to the output
      (default: sqrt(2. / features), see Algorithm 1 of [3]).
    activation: activation function to apply to the output.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    seed: random seed for generating random features (default: 0). This will
      override the external RNGs.
    dtype: the dtype of the computation (default: float32).
  """
  features: int
  feature_scale: Optional[jnp.float32] = 1.
  activation: Callable[[Array], Array] = default_rbf_activation
  kernel_init: Initializer = default_rbf_kernel_init
  bias_init: Initializer = default_rbf_bias_init
  seed: int = 0
  dtype: Dtype = jnp.float32
  collection_name: str = 'random_features'

  def setup(self):
    # Defines the random number generator.
    self.rng = random.PRNGKey(self.seed)

    # Processes random feature scale.
    self._feature_scale = self.feature_scale
    if self._feature_scale is None:
      self._feature_scale = jnp.sqrt(2. / self.features)
    self._feature_scale = jnp.asarray(self._feature_scale, dtype=self.dtype)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies random feature transformation along the last dimension of inputs.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    # Initializes variables.
    input_dim = inputs.shape[-1]

    kernel_rng, bias_rng = random.split(self.rng, num=2)
    kernel_shape = (input_dim, self.features)

    kernel = self.variable(self.collection_name, 'kernel', self.kernel_init,
                           kernel_rng, kernel_shape, self.dtype)
    bias = self.variable(self.collection_name, 'bias', self.bias_init,
                         bias_rng, (self.features,), self.dtype)

    # Specifies multiplication dimension.
    contracting_dims = ((inputs.ndim - 1,), (0,))
    batch_dims = ((), ())

    # Performs forward pass.
    inputs = jnp.asarray(inputs, self.dtype)
    outputs = lax.dot_general(inputs, kernel.value,
                              (contracting_dims, batch_dims))
    outputs = outputs + jnp.broadcast_to(bias.value, outputs.shape)

    return self._feature_scale * self.activation(outputs)


class LaplaceRandomFeatureCovariance(nn.Module):
  """Computes the Gaussian Process covariance using Laplace method.

  Attributes:
    hidden_features: the number of random fourier features.
    ridge_penalty: Initial Ridge penalty to weight covariance matrix. This value
      is used to stablize the eigenvalues of weight covariance estimate so that
      the matrix inverse can be computed for Cov = inv(t(X) @ X + s * I). The
      ridge factor s cannot be too large since otherwise it will dominate the
      t(X) * X term and make covariance estimate not meaningful.
    momentum: A discount factor used to compute the moving average for posterior
      precision matrix. Analogous to the momentum factor in batch normalization.
      If `None` then update covariance matrix using a naive sum without
      momentum, which is desirable if the goal is to compute the exact
      covariance matrix by passing through data once (say in the final epoch).
      In this case, make sure to reset the precision matrix variable between
      epochs by replacing it with self.initial_precision_matrix().
    likelihood: The likelihood to use for computing Laplace approximation for
      the covariance matrix. Can be one of ('binary_logistic', 'poisson',
      'gaussian').
  """
  hidden_features: int
  ridge_penalty: float = 1.
  momentum: Optional[float] = None
  likelihood: str = 'gaussian'
  collection_name: str = 'laplace_covariance'
  dtype: Dtype = jnp.float32

  def setup(self):
    if self.momentum is not None:
      if self.momentum < 0. or self.momentum > 1.:
        raise ValueError(f'`momentum` must be between (0, 1). '
                         f'Got {self.momentum}.')

    if self.likelihood not in SUPPORTED_LIKELIHOOD:
      raise ValueError(f'"likelihood" must be one of {SUPPORTED_LIKELIHOOD}, '
                       f'got {self.likelihood}.')

  @nn.compact
  def __call__(self,
               gp_features: Array,
               gp_logits: Optional[Array] = None,
               diagonal_only: bool = True) -> Optional[Array]:
    """Updates the precision matrix and computes the predictive covariance.

    NOTE:
    The precision matrix will be updated only during training (i.e., when
    `self.collection_name` are in the list of mutable variables). The covariance
    matrix will be computed only during inference to avoid repeated calls to the
    (expensive) `linalg.inv` op.

    Args:
      gp_features: The nd-array of random fourier features, shape (batch_size,
        ..., hidden_features).
      gp_logits: The nd-array of predictive logits, shape (batch_size, ...,
        logit_dim). Cannot be None if self.likelihood is not `gaussian`.
      diagonal_only: Whether to return only the diagonal elements of the
        predictive covariance matrix (i.e., the predictive variance).

    Returns:
      The predictive variances of shape (batch_size, ) if diagonal_only=True,
      otherwise the predictive covariance matrix of shape
      (batch_size, batch_size).
    """
    gp_features = jnp.asarray(gp_features, self.dtype)

    # Flatten GP features and logits to 2-d, by doing so we treat all the
    # non-final dimensions as the batch dimensions.
    gp_features = jnp.reshape(gp_features, [-1, self.hidden_features])

    if gp_logits is not None:
      gp_logits = jnp.asarray(gp_logits, self.dtype)
      gp_logits = jnp.reshape(gp_logits, [gp_features.shape[0], -1])

    precision_matrix = self.variable(self.collection_name, 'precision_matrix',
                                     lambda: self.initial_precision_matrix())  # pylint: disable=unnecessary-lambda

    # Updates the precision matrix during training.
    initializing = self.is_mutable_collection('params')
    training = self.is_mutable_collection(self.collection_name)

    if training and not initializing:
      precision_matrix.value = self.update_precision_matrix(
          gp_features, gp_logits, precision_matrix.value)

    # Computes covariance matrix during inference.
    if not training:
      return self.compute_predictive_covariance(gp_features, precision_matrix,
                                                diagonal_only)

  def initial_precision_matrix(self):
    """Returns the initial diagonal precision matrix."""
    return jnp.eye(self.hidden_features, dtype=self.dtype) * self.ridge_penalty

  def update_precision_matrix(self, gp_features: Array,
                              gp_logits: Optional[Array],
                              precision_matrix: Array) -> Array:
    """Updates precision matrix given a new batch.

    Args:
      gp_features: random features from the new batch, shape (batch_size,
        hidden_features)
      gp_logits: predictive logits from the new batch, shape (batch_size,
        logit_dim). Currently only logit_dim=1 is supported.
      precision_matrix: the current precision matrix, shape (hidden_features,
        hidden_features).

    Returns:
      Updated precision matrix, shape (hidden_features, hidden_features).

    Raises:
      (ValueError) If the logit is None or not univariate when likelihood is
        not Gaussian.
    """
    if self.likelihood != 'gaussian':
      if gp_logits is None:
        raise ValueError(
            f'`gp_logits` cannot be None when likelihood=`{self.likelihood}`')

      if gp_logits.ndim > 1 and gp_logits.shape[-1] != 1:
        raise ValueError(
            f'likelihood `{self.likelihood}` only support univariate logits. '
            f'Got logits dimension: {gp_logits.shape[-1]}')

    # Computes precision matrix within new batch.
    if self.likelihood == 'binary_logistic':
      prob = nn.sigmoid(gp_logits)
      prob_multiplier = prob * (1. - prob)
    elif self.likelihood == 'poisson':
      prob_multiplier = jnp.exp(gp_logits)
    else:
      prob_multiplier = 1.

    gp_features_adj = jnp.sqrt(prob_multiplier) * gp_features
    batch_prec_mat = jnp.matmul(jnp.transpose(gp_features_adj), gp_features_adj)

    # Updates precision matrix.
    if self.momentum is None:
      # Performs exact update without momentum.
      precision_matrix_updated = precision_matrix + batch_prec_mat
    else:
      batch_size = gp_features.shape[0]
      precision_matrix_updated = (
          self.momentum * precision_matrix +
          (1 - self.momentum) * batch_prec_mat / batch_size)
    return precision_matrix_updated

  def compute_predictive_covariance(self, gp_features: Array,
                                    precision_matrix: nn.Variable,
                                    diagonal_only: bool) -> Array:
    """Computes the predictive covariance.

    Approximates the Gaussian process posterior using random features.
    Given training random feature Phi_tr (num_train, num_hidden) and testing
    random feature Phi_ts (batch_size, num_hidden). The predictive covariance
    matrix is computed as (assuming Gaussian likelihood):

    s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

    where s is the ridge factor to be used for stablizing the inverse, and I is
    the identity matrix with shape (num_hidden, num_hidden).

    Args:
      gp_features: the random feature of testing data to be used for computing
        the covariance matrix. Shape (batch_size, gp_hidden_size).
      precision_matrix: the model's precision matrix.
      diagonal_only: whether to return only the diagonal elements of the
        predictive covariance matrix (i.e., the predictive variances).

    Returns:
      The predictive variances of shape (batch_size, ) if diagonal_only=True,
      otherwise the predictive covariance matrix of shape
      (batch_size, batch_size).
    """
    precision_matrix_inv = jnp.linalg.inv(precision_matrix.value)
    cov_feature_product = jnp.matmul(precision_matrix_inv,
                                     jnp.transpose(gp_features))

    if diagonal_only:
      # Compute diagonal element only, shape (batch_size, ).
      # Using the identity diag(A @ B) = col_sum(A * tr(B)).
      gp_covar = jnp.sum(
          gp_features * jnp.transpose(cov_feature_product), axis=-1)
    else:
      # Compute full covariance matrix, shape (batch_size, batch_size).
      gp_covar = jnp.matmul(gp_features, cov_feature_product)

    return self.ridge_penalty * gp_covar


class MCSigmoidDenseFASNGP(nn.Module):
  """Heteroscedastic SNGP for data with sigmoid output activation.

  Output layer which combines the benefits of the heteroscedastic
  (https://arxiv.org/abs/2105.10305) and SNGP (https://arxiv.org/abs/2006.10108)
  methods. Assumes spectral normalization is applied to network producing
  `inputs` to the __call__ method.

  Attributes:
      num_outputs: Number of outputs for classification task.
      num_factors: Number of factors to use in approximation to full rank
        covariance matrix.
      temperature: The softmax temperature.
      parameter_efficient: Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If False then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      share_samples_across_batch: If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference setting = True may solve.
      logits_only: If True, only return the logits from the __call__ method.
      return_locs: If True, return the location parameter of the Gaussian
        latent variable in place of the `logits`.
      eps: Clip probabilities into [eps, 1.0] before applying log.
      het_var_weight: Weighting on the heteroscedastic variance when computing
        samples from the Gaussian latent variable.
      sngp_var_weight: Weighting on the GP variance when computing samples from
        the Gaussian latent variable.
      hidden_features: Number of features for Random Fourier Feature GP
        approximation.
      normalize_input: Whether to normalize the input for the GP layer.
      norm_kwargs: Normalization keywords for the GP layer.
      hidden_kwargs: Hidden layer keywords for the GP layer.
      output_kwargs: Output keywords for the GP layer.
      covmat_kwargs: Covariance matrix keywords for the GP layer.
  """
  num_outputs: int
  num_factors: int  # set num_factors = 0 for diagonal method
  temperature: float = 1.0
  parameter_efficient: bool = False
  train_mc_samples: int = 1000
  test_mc_samples: int = 1000
  share_samples_across_batch: bool = False
  logits_only: bool = False
  return_locs: bool = False
  eps: float = 1e-7
  het_var_weight: float = 1.0
  sngp_var_weight: float = 0.0

  hidden_features: int = 1024
  normalize_input: bool = True

  # Optional keyword arguments.
  norm_kwargs: Mapping[str, Any] = default_kwarg_dict()
  hidden_kwargs: Mapping[str, Any] = default_kwarg_dict()
  output_kwargs: Mapping[str, Any] = default_kwarg_dict()
  covmat_kwargs: Mapping[str, Any] = default_kwarg_dict()

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = nn.Dense(
          self.num_outputs * self.num_factors, name='scale_layer')

    self._loc_layer = RandomFeatureGaussianProcess(
        features=self.num_outputs,
        hidden_features=self.hidden_features,
        normalize_input=self.normalize_input,
        norm_kwargs=self.norm_kwargs,
        hidden_kwargs=self.hidden_kwargs,
        output_kwargs=self.output_kwargs,
        covmat_kwargs=self.covmat_kwargs,
        name='loc_layer')
    self._diag_layer = nn.Dense(self.num_outputs, name='diag_layer')

  def _compute_loc_param(self, inputs: Array) -> Array:
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: The input to the heteroscedastic output layer.

    Returns:
      Array of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs: Array, covmat_sngp: Array,
                           training: int) -> Tuple[Array, Array]:
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: The input to the heteroscedastic output layer.
      covmat_sngp: GP output layer covariance matrix.
      training: in training mode or not.

    Returns:
      2-Tuple of Array of shape
      ([batch_size, num_classes * max(num_factors, 1)],
      [batch_size, num_classes]).
    """
    if self.parameter_efficient or self.num_factors <= 0:
      low_rank = inputs
      diag = jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO
    else:
      low_rank = self._scale_layer(inputs)
      diag = jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO

    initializing = self.is_mutable_collection('params')
    if training or initializing:
      diag_comp = diag
    else:
      # assume diagonal_only=True
      sngp_marginal_vars = jnp.expand_dims(covmat_sngp, -1)
      diag_comp = jnp.sqrt(self.het_var_weight * jnp.square(diag) +
                           self.sngp_var_weight * sngp_marginal_vars)

    return low_rank, diag_comp

  def _compute_diagonal_noise_samples(self, diag_scale: Array,
                                      num_samples: int) -> Array:
    """Computes samples of the diagonal elements logit noise.

    Args:
      diag_scale: Array of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.

    Returns:
      Array. Logit noise samples of shape:
        [batch_size, num_samples, num_outputs].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = diag_scale.shape[0]

    key = self.make_rng('diag_noise_samples')
    return jnp.expand_dims(diag_scale, 1) * jax.random.normal(
        key, shape=(samples_per_batch, num_samples, 1))

  def _compute_standard_normal_samples(self, factor_loadings: Array,
                                       num_samples: int) -> Array:
    """Utility that computes samples from a standard normal distribution.

    Args:
      factor_loadings: Array of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.

    Returns:
      Array. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = factor_loadings.shape[0]

    key = self.make_rng('standard_norm_noise_samples')
    standard_normal_samples = jax.random.normal(
        key, shape=(samples_per_batch, num_samples, self.num_factors))

    if self.share_samples_across_batch:
      standard_normal_samples = jnp.tile(standard_normal_samples,
                                         [factor_loadings.shape[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale: Tuple[Array, Array],
                             num_samples: int) -> Array:
    """Utility function that computes additive noise samples.

    Args:
      scale: Tuple of Array of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.

    Returns:
      Array. Logit noise samples of shape:
        [batch_size, num_samples, num_outputs].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples)

    if self.num_factors > 0:
      # Now compute the factors
      standard_normal_samples = self._compute_standard_normal_samples(
          factor_loadings, num_samples)

      if self.parameter_efficient:
        res = self._scale_layer_homoscedastic(standard_normal_samples)
        res *= jnp.expand_dims(
            self._scale_layer_heteroscedastic(factor_loadings), 1)
      else:
        # reshape scale vector into factor loadings matrix
        factor_loadings = jnp.reshape(factor_loadings,
                                      [-1, self.num_outputs, self.num_factors])

        # transform standard normal into ~ full rank covariance Gaussian samples
        res = jnp.einsum('ijk,iak->iaj',
                         factor_loadings, standard_normal_samples)
      return res + diag_noise_samples
    return diag_noise_samples

  def _compute_mc_samples(self, locs: Array, scale: Array,
                          num_samples: int) -> Array:
    """Utility function that computes Monte-Carlo samples (using sigmoid).

    Args:
      locs: Array of shape [batch_size, total_mc_samples, num_outputs].
        Location parameters of the distributions to be sampled.
      scale: Array of shape [batch_size, total_mc_samples, num_outputs].
        Scale parameters of the distributions to be sampled.
      num_samples: Number of Monte-Carlo samples to take.

    Returns:
      Array of shape [batch_size, num_samples, num_outputs]. Average over the
        MC samples.
    """
    locs = jnp.expand_dims(locs, axis=1)

    noise_samples = self._compute_noise_samples(scale, num_samples)

    latents = locs + noise_samples
    samples = jax.nn.sigmoid(latents / self.temperature)

    return jnp.mean(samples, axis=1)

  @nn.compact
  def __call__(self, inputs: Array, training: int = True) -> Union[
      Tuple[Array, Array], Tuple[Array, Array, Array, Array]]:
    """Computes predictive and log predictive distributions.

    Uses Monte Carlo estimate of sigmoid approximation to HetSNGP model to
    compute predictive distribution.

    Args:
      inputs: The input to the heteroscedastic output layer.
      training: Whether we are training or not.

    Returns:
      Tuple of Array: (logits, covmat_sngp) if logits_only = True. Otherwise,
      tuple of (logits, covmat_sngp, log_probs, probs). Logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      sigmoid cross-entropy loss function.
    """
    # return_random_features set to False, so guaranteed to return 2-tuple
    locs, covmat_sngp = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none,unbalanced-tuple-unpacking
    # guaranteed to return 2-tuple due to scale_layer construction
    scale = self._compute_scale_param(inputs, covmat_sngp, training)  # pylint: disable=assignment-from-none

    if training:
      total_mc_samples = self.train_mc_samples
    else:
      total_mc_samples = self.test_mc_samples

    probs_mean = self._compute_mc_samples(locs, scale, total_mc_samples)

    probs_mean = jnp.clip(probs_mean, a_min=self.eps)
    log_probs = jnp.log(probs_mean)

    # inverse sigmoid
    probs_mean = jnp.clip(probs_mean, a_min=self.eps, a_max=1.0 - self.eps)
    logits = log_probs - jnp.log(1.0 - probs_mean)

    if self.return_locs:
      logits = locs

    if self.logits_only:
      return logits, covmat_sngp

    return logits, covmat_sngp, log_probs, probs_mean


class MCSigmoidDenseFASNGPBE(MCSigmoidDenseFASNGP):
  """Heteroscedastic SNGP + BE for data with sigmoid output activation.
  """

  ens_size: int = 1

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = dense.DenseBatchEnsemble(
          self.num_outputs,
          ens_size=self.ens_size,
          name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = dense.DenseBatchEnsemble(
          self.num_outputs,
          ens_size=self.ens_size,
          name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = dense.DenseBatchEnsemble(
          self.num_outputs * self.num_factors,
          ens_size=self.ens_size,
          name='scale_layer')

    self._loc_layer = RandomFeatureGaussianProcess(
        features=self.num_outputs,
        hidden_features=self.hidden_features,
        normalize_input=self.normalize_input,
        norm_kwargs=self.norm_kwargs,
        hidden_kwargs=self.hidden_kwargs,
        output_kwargs=self.output_kwargs,
        covmat_kwargs=self.covmat_kwargs,
        name='loc_layer')
    self._diag_layer = dense.DenseBatchEnsemble(self.num_outputs,
                                                ens_size=self.ens_size,
                                                name='diag_layer')
