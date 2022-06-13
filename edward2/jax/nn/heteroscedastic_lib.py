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

"""Library of methods to compute heteroscedastic classification predictions."""

from typing import Iterable, Callable, Optional

from edward2.jax.nn import dense
import flax.linen as nn
import jax
import jax.numpy as jnp

DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]

MIN_SCALE_MONTE_CARLO = 1e-3
TEMPERATURE_LOWER_BOUND = 0.3
TEMPERATURE_UPPER_BOUND = 3.0


def compute_temperature(pre_sigmoid_temperature: jnp.ndarray,
                        lower: Optional[float],
                        upper: Optional[float]) -> jnp.ndarray:
  """Compute the temperature based on the sigmoid parametrization."""
  lower = lower if lower is not None else TEMPERATURE_LOWER_BOUND
  upper = upper if upper is not None else TEMPERATURE_UPPER_BOUND
  temperature = jax.nn.sigmoid(pre_sigmoid_temperature)
  return (upper - lower) * temperature + lower


class MCSoftmaxDenseFA(nn.Module):
  """Softmax and factor analysis approx to heteroscedastic predictions.

  if we assume:
  u ~ N(mu(x), sigma(x))
  and
  y = softmax(u / temperature)

  we can do a low rank approximation of sigma(x) the full rank matrix as:
  eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
  u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
  where V(x) is a matrix of dimension [num_classes, R=num_factors]
  and d(x) is a vector of dimension [num_classes, 1]
  num_factors << num_classes => approx to sampling ~ N(mu(x), sigma(x))
  """

  num_classes: int
  num_factors: int  # set num_factors = 0 for diagonal method
  temperature: float = 1.0
  parameter_efficient: bool = False
  train_mc_samples: int = 1000
  test_mc_samples: int = 1000
  share_samples_across_batch: bool = False
  logits_only: bool = False
  return_locs: bool = False
  eps: float = 1e-7
  tune_temperature: bool = False
  temperature_lower_bound: Optional[float] = None
  temperature_upper_bound: Optional[float] = None

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = nn.Dense(
          self.num_classes, name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = nn.Dense(
          self.num_classes, name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = nn.Dense(
          self.num_classes * self.num_factors, name='scale_layer')

    self._loc_layer = nn.Dense(self.num_classes, name='loc_layer')
    self._diag_layer = nn.Dense(self.num_classes, name='diag_layer')

    if self.tune_temperature:
      # A zero-initialization means the midpoint of the temperature interval
      # after applying the sigmoid transformation.
      self._pre_sigmoid_temperature = self.param('pre_sigmoid_temperature',
                                                 nn.initializers.zeros, (1,))
    else:
      self._pre_sigmoid_temperature = None

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape
      ([batch_size, num_classes * max(num_factors, 1)],
      [batch_size, num_classes]).
    """
    if self.parameter_efficient or self.num_factors <= 0:
      return (inputs,
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = diag_scale.shape[0]

    key = self.make_rng('diag_noise_samples')
    return jnp.expand_dims(diag_scale, 1) * jax.random.normal(
        key, shape=(samples_per_batch, num_samples, 1))

  def _compute_standard_normal_samples(self, factor_loadings, num_samples):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
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

  def _compute_noise_samples(self, scale, num_samples):
    """Utility function to compute additive noise samples.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
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
                                      [-1, self.num_classes, self.num_factors])

        # transform standard normal into ~ full rank covariance Gaussian samples
        res = jnp.einsum('ijk,iak->iaj',
                         factor_loadings, standard_normal_samples)
      return res + diag_noise_samples
    return diag_noise_samples

  def get_temperature(self):
    if self.tune_temperature:
      return compute_temperature(
          self._pre_sigmoid_temperature,
          lower=self.temperature_lower_bound,
          upper=self.temperature_upper_bound)
    else:
      return self.temperature

  def _compute_mc_samples(self, locs, scale, num_samples):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = jnp.expand_dims(locs, axis=1)

    noise_samples = self._compute_noise_samples(scale, num_samples)

    latents = locs + noise_samples
    samples = jax.nn.softmax(latents / self.get_temperature())

    return jnp.mean(samples, axis=1)

  @nn.compact
  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distributions.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). logits can be
      used with the standard softmax cross-entropy loss function.
    """
    locs = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none
    scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

    if training:
      total_mc_samples = self.train_mc_samples
    else:
      total_mc_samples = self.test_mc_samples

    probs_mean = self._compute_mc_samples(locs, scale, total_mc_samples)

    probs_mean = jnp.clip(probs_mean, a_min=self.eps)
    log_probs = jnp.log(probs_mean)
    logits = log_probs

    if self.return_locs:
      logits = locs

    if self.logits_only:
      return logits

    return logits, log_probs, probs_mean


class MCSigmoidDenseFA(nn.Module):
  """Sigmoid and factor analysis approx to heteroscedastic predictions.

  if we assume:
  u ~ N(mu(x), sigma(x))
  and
  y = sigmoid(u / temperature)

  we can do a low rank approximation of sigma(x) the full rank matrix as:
  eps_R ~ N(0, I_R), eps_K ~ N(0, identity_K)
  u = mu(x) + matmul(V(x), e) + d(x) * e_d
  where A(x) is a matrix of dimension [num_outputs, R=num_factors]
  and d(x) is a vector of dimension [num_outputs, 1]
  num_factors << num_outputs => approx to sampling ~ N(mu(x), sigma(x)).
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
  tune_temperature: bool = False
  temperature_lower_bound: Optional[float] = None
  temperature_upper_bound: Optional[float] = None

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = nn.Dense(
          self.num_outputs, name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = nn.Dense(
          self.num_outputs * self.num_factors, name='scale_layer')

    self._loc_layer = nn.Dense(self.num_outputs, name='loc_layer')
    self._diag_layer = nn.Dense(self.num_outputs, name='diag_layer')

    if self.tune_temperature:
      # A zero-initialization means the midpoint of the temperature interval
      # after applying the sigmoid transformation.
      self._pre_sigmoid_temperature = self.param('pre_sigmoid_temperature',
                                                 nn.initializers.zeros, (1,))
    else:
      self._pre_sigmoid_temperature = None

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape
      ([batch_size, num_classes * max(num_factors, 1)],
      [batch_size, num_classes]).
    """
    if self.parameter_efficient or self.num_factors <= 0:
      return (inputs,
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              jax.nn.softplus(self._diag_layer(inputs)) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self.share_samples_across_batch:
      samples_per_batch = 1
    else:
      samples_per_batch = diag_scale.shape[0]

    key = self.make_rng('diag_noise_samples')
    return jnp.expand_dims(diag_scale, 1) * jax.random.normal(
        key, shape=(samples_per_batch, num_samples, 1))

  def _compute_standard_normal_samples(self, factor_loadings, num_samples):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
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

  def _compute_noise_samples(self, scale, num_samples):
    """Utility function to compute additive noise samples.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
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

  def get_temperature(self):
    if self.tune_temperature:
      return compute_temperature(
          self._pre_sigmoid_temperature,
          lower=self.temperature_lower_bound,
          upper=self.temperature_upper_bound)
    else:
      return self.temperature

  def _compute_mc_samples(self, locs, scale, num_samples):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = jnp.expand_dims(locs, axis=1)

    noise_samples = self._compute_noise_samples(scale, num_samples)

    latents = locs + noise_samples
    samples = jax.nn.sigmoid(latents / self.get_temperature())

    return jnp.mean(samples, axis=1)

  @nn.compact
  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distributions.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). Logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      sigmoid cross-entropy loss function.
    """
    locs = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none
    scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

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
      return logits

    return logits, log_probs, probs_mean


class MCSoftmaxDenseFABE(MCSoftmaxDenseFA):
  """Softmax and factor analysis approx to heteroscedastic + BatchEnsemble.
  """

  ens_size: int = 1
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = nn.initializers.lecun_normal()

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = dense.DenseBatchEnsemble(
          self.num_classes,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = dense.DenseBatchEnsemble(
          self.num_classes,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = dense.DenseBatchEnsemble(
          self.num_classes * self.num_factors,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer')

    self._loc_layer = dense.DenseBatchEnsemble(self.num_classes,
                                               ens_size=self.ens_size,
                                               name='loc_layer')
    self._diag_layer = dense.DenseBatchEnsemble(self.num_classes,
                                                ens_size=self.ens_size,
                                                name='diag_layer')


class MCSigmoidDenseFABE(MCSigmoidDenseFA):
  """Sigmoid and factor analysis approx to heteroscedastic + BatchEnsemble.
  """

  ens_size: int = 1
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = nn.initializers.lecun_normal()

  def setup(self):
    if self.parameter_efficient:
      self._scale_layer_homoscedastic = dense.DenseBatchEnsemble(
          self.num_outputs,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer_homoscedastic')
      self._scale_layer_heteroscedastic = dense.DenseBatchEnsemble(
          self.num_outputs,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer_heteroscedastic')
    elif self.num_factors > 0:
      self._scale_layer = dense.DenseBatchEnsemble(
          self.num_outputs * self.num_factors,
          ens_size=self.ens_size,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          kernel_init=self.kernel_init,
          name='scale_layer')

    self._loc_layer = dense.DenseBatchEnsemble(self.num_outputs,
                                               ens_size=self.ens_size,
                                               alpha_init=self.alpha_init,
                                               gamma_init=self.gamma_init,
                                               kernel_init=self.kernel_init,
                                               name='loc_layer')
    self._diag_layer = dense.DenseBatchEnsemble(self.num_outputs,
                                                ens_size=self.ens_size,
                                                alpha_init=self.alpha_init,
                                                gamma_init=self.gamma_init,
                                                kernel_init=self.kernel_init,
                                                name='diag_layer')
