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

"""Uncertainty-based dense layers."""

import math
from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import initializers
from edward2.tensorflow import random_variable
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow as tf
import tensorflow_probability as tfp

LAMBDA_TYPE = ('l2_kernel', 'l2_bias', 'dr')  # used by HyperBatchEnsemble


@utils.add_weight
class DenseReparameterization(tf.python.keras.layers.Dense):
  """Bayesian densely-connected layer estimated via reparameterization.

  The layer computes a variational Bayesian approximation to the distribution
  over densely-connected layers,

  ```
  p(outputs | inputs) = int dense(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               **kwargs):
    super().__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.python.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.python.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super().call(*args, **kwargs)


class DenseDVI(DenseReparameterization):
  """Densely-connected layer with deterministic VI (Wu et al., 2018).

  This layer computes a variational inference approximation via first and second
  moments. It is accurate if the kernel and bias initializers return factorized
  normal random variables and the number of units is sufficiently large. The
  advantage is that the forward pass is deterministic, reducing variance of
  gradients during training. The disadvantage is an O(features^2*units) compute
  and O(features^2 + features*units) memory complexity. In comparison,
  DenseReparameterization has O(features*units) compute and memory complexity.

  #### Examples

  Below implements deterministic variational inference for Bayesian
  feedforward network regression. We use the exact expected log-likelihood from
  Wu et al. (2018), Eq. 8. Assume 2-D real-valued tensors of `features` and
  `labels` of shapes `[batch_size, num_features]` and `[batch_size, 1]`
  respectively.

  ```python
  model = tf.python.keras.Sequential([
      ed.layers.DenseDVI(256, activation=tf.nn.relu),
      ed.layers.DenseDVI(256, activation=tf.nn.relu),
      ed.layers.DenseDVI(1, activation=None),
  ])

  # Run training loop.
  num_steps = 1000
  for _ in range(num_steps):
    with tf.GradientTape() as tape:
      locs = model(features)
      nll = 0.5 * tf.reduce_mean(locs.distribution.variance() +
                                 (labels - locs.distribution.mean())**2)
      kl = sum(model.losses) / total_dataset_size
      loss = nll + kl
    gradients = tape.gradient(loss, model.variables)  # use any optimizer here
  ```

  For evaluation, feed in data and use, e.g., `predictions.distribution.mean()`
  to make predictions via the posterior predictive distribution.

  ```python
  predictions = ed.Normal(loc=locs.distribution.mean(),
                          scale=locs.distribution.variance() + 1.)
  ```
  """

  def call(self, inputs):
    if (not isinstance(inputs, random_variable.RandomVariable) and
        not isinstance(self.kernel, random_variable.RandomVariable) and
        not isinstance(self.bias, random_variable.RandomVariable)):
      return super().call(inputs)
    self.call_weights()
    inputs_mean, inputs_variance, inputs_covariance = get_moments(inputs)
    kernel_mean, kernel_variance, _ = get_moments(self.kernel)
    if self.use_bias:
      bias_mean, _, bias_covariance = get_moments(self.bias)

    # E[outputs] = E[inputs] * E[kernel] + E[bias]
    mean = tf.tensordot(inputs_mean, kernel_mean, [[-1], [0]])
    if self.use_bias:
      mean = tf.nn.bias_add(mean, bias_mean)

    # Cov = E[inputs**2] Cov(kernel) + E[W]^T Cov(inputs) E[W] + Cov(bias)
    # For first term, assume Cov(kernel) = 0 on off-diagonals so we only
    # compute diagonal term.
    covariance_diag = tf.tensordot(inputs_variance + inputs_mean**2,
                                   kernel_variance, [[-1], [0]])
    # Compute quadratic form E[W]^T Cov E[W] from right-to-left. First is
    #  [..., features, features], [features, units] -> [..., features, units].
    cov_w = tf.tensordot(inputs_covariance, kernel_mean, [[-1], [0]])
    # Next is [..., features, units], [features, units] -> [..., units, units].
    w_cov_w = tf.tensordot(cov_w, kernel_mean, [[-2], [0]])
    covariance = w_cov_w
    if self.use_bias:
      covariance += bias_covariance
    covariance = tf.linalg.set_diag(
        covariance, tf.linalg.diag_part(covariance) + covariance_diag)

    if self.activation in (tf.python.keras.activations.relu, tf.nn.relu):
      # Compute activation's moments with variable names from Wu et al. (2018).
      variance = tf.linalg.diag_part(covariance)
      scale = tf.sqrt(variance)
      mu = mean / (scale + tf.python.keras.backend.epsilon())
      mean = scale * soft_relu(mu)

      pairwise_variances = (tf.expand_dims(variance, -1) *
                            tf.expand_dims(variance, -2))  # [..., units, units]
      rho = covariance / tf.sqrt(pairwise_variances +
                                 tf.python.keras.backend.epsilon())
      rho = tf.clip_by_value(rho,
                             -1. / (1. + tf.python.keras.backend.epsilon()),
                             1. / (1. + tf.python.keras.backend.epsilon()))
      s = covariance / (rho + tf.python.keras.backend.epsilon())
      mu1 = tf.expand_dims(mu, -1)  # [..., units, 1]
      mu2 = tf.linalg.matrix_transpose(mu1)  # [..., 1, units]
      a = (soft_relu(mu1) * soft_relu(mu2) +
           rho * tfp.distributions.Normal(0., 1.).cdf(mu1) *
           tfp.distributions.Normal(0., 1.).cdf(mu2))
      gh = tf.asinh(rho)
      bar_rho = tf.sqrt(1. - rho**2)
      gr = gh + rho / (1. + bar_rho)
      # Include numerically stable versions of gr and rho when multiplying or
      # dividing them. The sign of gr*rho and rho/gr is always positive.
      safe_gr = tf.abs(gr) + 0.5 * tf.python.keras.backend.epsilon()
      safe_rho = tf.abs(rho) + tf.python.keras.backend.epsilon()
      exp_negative_q = gr / (2. * math.pi) * tf.exp(
          -safe_rho / (2. * safe_gr * (1 + bar_rho)) +
          (gh - rho) / (safe_gr * safe_rho) * mu1 * mu2)
      covariance = s * (a + exp_negative_q)
    elif self.activation not in (tf.python.keras.activations.linear, None):
      raise NotImplementedError('Activation is {}. Deterministic variational '
                                'inference is only available if activation is '
                                'ReLU or None.'.format(self.activation))

    return generated_random_variables.MultivariateNormalFullCovariance(
        mean, covariance)


def get_moments(x):
  """Gets first and second moments of input."""
  if isinstance(x, random_variable.RandomVariable):
    mean = x.distribution.mean()
    variance = x.distribution.variance()
    try:
      covariance = x.distribution.covariance()
    except NotImplementedError:
      covariance = tf.zeros(x.shape.concatenate(x.shape[-1]), dtype=x.dtype)
      covariance = tf.linalg.set_diag(covariance, variance)
  else:
    mean = x
    variance = tf.zeros_like(x)
    covariance = tf.zeros(x.shape.concatenate(x.shape[-1]), dtype=x.dtype)
  return mean, variance, covariance


def soft_relu(x):
  return (tfp.distributions.Normal(0., 1.).prob(x) +
          x * tfp.distributions.Normal(0., 1.).cdf(x))


class DenseFlipout(DenseReparameterization):
  """Bayesian densely-connected layer estimated via Flipout (Wen et al., 2018).

  The layer computes a variational Bayesian approximation to the distribution
  over densely-connected layers,

  ```
  p(outputs | inputs) = int dense(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` random variable must be able
  to decompose as a sum of its mean and a perturbation distribution; the
  perturbation distribution must be independent across weight elements and
  symmetric around zero (for example, a fully factorized Gaussian).
  """

  def call(self, inputs):
    if not isinstance(self.kernel, random_variable.RandomVariable):
      return super().call(inputs)
    self.call_weights()
    input_shape = tf.shape(inputs)
    output_shape = tf.concat([input_shape[:-1], [self.units]], 0)
    sign_input = tf.cast(2 * tf.random.uniform(input_shape,
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32) - 1,
                         inputs.dtype)
    sign_output = tf.cast(2 * tf.random.uniform(output_shape,
                                                minval=0,
                                                maxval=2,
                                                dtype=inputs.dtype) - 1,
                          inputs.dtype)
    kernel_mean = self.kernel.distribution.mean()
    perturbation = self.kernel - kernel_mean
    if inputs.shape.ndims <= 2:
      outputs = tf.matmul(inputs, kernel_mean)
      outputs += tf.matmul(inputs * sign_input, perturbation) * sign_output
    else:
      outputs = tf.tensordot(inputs, kernel_mean, [[-1], [0]])
      outputs += tf.tensordot(inputs * sign_input,
                              perturbation,
                              [[-1], [0]]) * sign_output
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs


class DenseVariationalDropout(DenseReparameterization):
  """Densely-connected layer with variational dropout (Kingma et al., 2015).

  Implementation follows the additive parameterization of
  Molchanov et al. (2017).
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               kernel_regularizer='log_uniform_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               **kwargs):
    super().__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def call(self, inputs, training=None):
    if not isinstance(self.kernel, random_variable.RandomVariable):
      return super().call(inputs)
    self.call_weights()
    if training is None:
      training = tf.python.keras.backend.learning_phase()

    def dropped_inputs():
      """Forward pass with dropout."""
      # Clip magnitude of dropout rate, where we get the dropout rate alpha from
      # the additive parameterization (Molchanov et al., 2017): for weight ~
      # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
      mean = self.kernel.distribution.mean()
      log_variance = tf.math.log(self.kernel.distribution.variance())
      log_alpha = log_variance - tf.math.log(tf.square(mean) +
                                             tf.python.keras.backend.epsilon())
      log_alpha = tf.clip_by_value(log_alpha, -8., 8.)
      log_variance = log_alpha + tf.math.log(tf.square(mean) +
                                             tf.python.keras.backend.epsilon())

      if inputs.shape.ndims <= 2:
        means = tf.matmul(inputs, mean)
        stddevs = tf.sqrt(
            tf.matmul(tf.square(inputs), tf.exp(log_variance)) +
            tf.python.keras.backend.epsilon())
      else:
        means = tf.tensordot(inputs, mean, [[-1], [0]])
        stddevs = tf.sqrt(
            tf.tensordot(tf.square(inputs), tf.exp(log_variance), [[-1], [0]]) +
            tf.python.keras.backend.epsilon())
      if self.use_bias:
        means = tf.nn.bias_add(means, self.bias)
      outputs = generated_random_variables.Normal(loc=means, scale=stddevs)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.python.keras.Dropout, only apply variational dropout if training
    # flag is True.
    training_value = utils.smart_constant_value(training)
    if training_value is not None:
      if training_value:
        return dropped_inputs()
      else:
        return super().call(inputs)
    return tf.cond(
        pred=training,
        true_fn=dropped_inputs,
        false_fn=lambda: super(DenseVariationalDropout, self).call(inputs))


class DenseHierarchical(DenseVariationalDropout):
  """Bayesian densely-connected layer with hierarchical distributions.

  The layer computes a variational Bayesian approximation to the distribution
  over densely-connected layers, and where the distribution over weights
  involves a hierarchical distribution with hidden unit noise coupling vectors
  of the kernel weight matrix (Louizos et al., 2017),

  ```
  p(outputs | inputs) = int dense(inputs; new_kernel, bias) p(kernel,
    local_scales, global_scale, bias) dkernel dlocal_scales dglobal_scale dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. The kernel is written in non-centered
  parameterization where

  ```
  new_kernel[i, j] = kernel[i, j] * local_scale[i] * global_scale.
  ```

  That is, there is "local" multiplicative noise which couples weights for each
  input neuron. There is also a "global" multiplicative noise which couples the
  entire weight matrix. By default, the weights are normally distributed and the
  local and global noises are half-Cauchy distributed; this makes the kernel a
  horseshoe distribution (Carvalho et al., 2009; Polson and Scott, 2012).

  The estimation uses local reparameterization to avoid sampling the full
  weights. Gradients with respect to the distributions' learnable parameters
  backpropagate via reparameterization. Minimizing cross-entropy plus the
  layer's losses performs variational minimum description length, i.e., it
  minimizes an upper bound to the negative marginal likelihood.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zero',
               local_scale_initializer='trainable_half_cauchy',
               global_scale_initializer='trainable_half_cauchy',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               local_scale_regularizer='half_cauchy_kl_divergence',
               global_scale_regularizer=regularizers.HalfCauchyKLDivergence(
                   scale=1e-5),
               activity_regularizer=None,
               local_scale_constraint='softplus',
               global_scale_constraint='softplus',
               **kwargs):
    self.local_scale_initializer = initializers.get(local_scale_initializer)
    self.global_scale_initializer = initializers.get(global_scale_initializer)
    self.local_scale_regularizer = regularizers.get(local_scale_regularizer)
    self.global_scale_regularizer = regularizers.get(global_scale_regularizer)
    self.local_scale_constraint = constraints.get(local_scale_constraint)
    self.global_scale_constraint = constraints.get(global_scale_constraint)
    super().__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        **kwargs)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    self.local_scale = self.add_weight(
        shape=(input_dim,),
        name='local_scale',
        initializer=self.local_scale_initializer,
        regularizer=self.local_scale_regularizer,
        constraint=self.local_scale_constraint)
    self.global_scale = self.add_weight(
        shape=(),
        name='global_scale',
        initializer=self.global_scale_initializer,
        regularizer=self.global_scale_regularizer,
        constraint=self.global_scale_constraint)
    super().build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.local_scale_initializer, tf.python.keras.layers.Layer):
      self.local_scale = self.local_scale_initializer(self.local_scale.shape,
                                                      self.dtype)
    if isinstance(self.global_scale_initializer, tf.python.keras.layers.Layer):
      self.global_scale = self.global_scale_initializer(self.global_scale.shape,
                                                        self.dtype)
    super().call_weights()

  def call(self, inputs, training=None):
    self.call_weights()
    # TODO(trandustin): Figure out what to set local/global scales to at test
    # time. Means don't exist for Half-Cauchy approximate posteriors.
    inputs *= self.local_scale[tf.newaxis, :] * self.global_scale
    return super().call(inputs, training=training)


class DenseBatchEnsemble(tf.python.keras.layers.Dense):
  """A batch ensemble dense layer."""

  def __init__(self,
               units,
               rank=1,
               ensemble_size=4,
               activation=None,
               use_bias=True,
               alpha_initializer='ones',
               gamma_initializer='ones',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        units=units,
        use_bias=False,
        activation=None,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=None,
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=None,
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=kernel_constraint,
        bias_constraint=None,
        **kwargs)
    self.rank = rank
    self.ensemble_size = ensemble_size
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.use_ensemble_bias = use_bias
    self.ensemble_activation = tf.python.keras.activations.get(activation)
    self.ensemble_bias_initializer = initializers.get(bias_initializer)
    self.ensemble_bias_regularizer = regularizers.get(bias_regularizer)
    self.ensemble_bias_constraint = constraints.get(bias_constraint)

  def _build_parent(self, input_shape):
    super().build(input_shape)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    super().build(input_shape)

    input_dim = input_shape[-1]
    if self.rank > 1:
      alpha_shape = [self.rank, self.ensemble_size, input_dim]
      gamma_shape = [self.rank, self.ensemble_size, self.units]
    else:
      alpha_shape = [self.ensemble_size, input_dim]
      gamma_shape = [self.ensemble_size, self.units]

    self.alpha = self.add_weight(
        name='alpha',
        shape=alpha_shape,
        initializer=self.alpha_initializer,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        name='gamma',
        shape=gamma_shape,
        initializer=self.gamma_initializer,
        trainable=True,
        dtype=self.dtype)
    if self.use_ensemble_bias:
      self.ensemble_bias = self.add_weight(
          name='ensemble_bias',
          shape=[self.ensemble_size, self.units],
          initializer=self.ensemble_bias_initializer,
          regularizer=self.ensemble_bias_regularizer,
          constraint=self.ensemble_bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.ensemble_bias = None
    self.built = True

  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    input_dim = self.alpha.shape[-1]
    examples_per_model = batch_size // self.ensemble_size

    # TODO(ywenxu): Merge the following two cases.
    if self.rank > 1:
      input_dim = self.alpha.shape[-1]
      alpha = tf.reshape(tf.tile(self.alpha, [1, 1, examples_per_model]),
                         [self.rank, batch_size, input_dim])
      gamma = tf.reshape(tf.tile(self.gamma, [1, 1, examples_per_model]),
                         [self.rank, batch_size, self.units])
      perturb_inputs = tf.expand_dims(inputs, 0) * alpha
      outputs = super().call(perturb_inputs)
      outputs = tf.reduce_sum(outputs * gamma, axis=0)
      outputs = tf.reshape(
          outputs, [self.ensemble_size, examples_per_model, -1])
    else:
      inputs = tf.reshape(
          inputs, [self.ensemble_size, examples_per_model, input_dim])
      alpha = tf.expand_dims(self.alpha, 1)
      gamma = tf.expand_dims(self.gamma, 1)
      perturb_inputs = inputs * alpha
      outputs = super().call(perturb_inputs) * gamma
    if self.use_ensemble_bias:
      bias = tf.expand_dims(self.ensemble_bias, 1)
      outputs += bias
    if self.ensemble_activation is not None:
      outputs = self.ensemble_activation(outputs)
    outputs = tf.reshape(outputs, [batch_size, self.units])
    return outputs

  def get_config(self):
    config = {
        'ensemble_size':
            self.ensemble_size,
        'ensemble_activation':
            tf.python.keras.activations.serialize(self.ensemble_activation),
        'use_ensemble_bias':
            self.use_ensemble_bias,
        'alpha_initializer':
            initializers.serialize(self.alpha_initializer),
        'gamma_initializer':
            initializers.serialize(self.gamma_initializer),
        'ensemble_bias_initializer':
            initializers.serialize(self.ensemble_bias_initializer),
        'ensemble_bias_regularizer':
            regularizers.serialize(self.ensemble_bias_regularizer),
        'ensemble_bias_constraint':
            constraints.serialize(self.ensemble_bias_constraint),
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


class _DenseBatchEnsembleNoFastWeights(DenseBatchEnsemble):
  """Version of DenseBatchEnsemble that does not create fast weights."""

  def __init__(self,
               units,
               rank=1,
               ensemble_size=4,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

    super().__init__(
        units,
        rank=rank,
        ensemble_size=ensemble_size,
        activation=activation,
        use_bias=use_bias,
        alpha_initializer=None,
        gamma_initializer=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

    self.alpha = None
    self.gamma = None

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    super()._build_parent(input_shape)
    if self.use_ensemble_bias:
      self.ensemble_bias = self.add_weight(
          name='ensemble_bias',
          shape=[self.ensemble_size, self.units],
          initializer=self.ensemble_bias_initializer,
          regularizer=self.ensemble_bias_regularizer,
          constraint=self.ensemble_bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.ensemble_bias = None
    self.built = True


class DenseHyperBatchEnsemble(tf.python.keras.layers.Layer):
  """Dense Hyper-BatchEnsemble layer that self-tunes hyperparameters.

  * W, W' of size (d_in, d_out)
  * b_j, b'_j of size (d_out,) for j in {1, ..., ensemble_size}.
  * e(lambdas) = [e1(lambdas), e2(lambdas)] of size (d_out, 1) and (d_out, 1)
  * input x of size (d_in,)

  The expression is:
    * the weights: x (W * (r_j s_j^T)) + x (e1(lambdas) * (W' u_j v_j^T)
    * the bias: b_j + (e2(lambdas) * b'_j)
  with j in {1, ..., ensemble_size}.

  The rank-1 perturbations are taken from ed.layers.DenseBatchEnsemble.

  Importantly, in https://arxiv.org/pdf/1903.03088.pdf, the e models are taken
  to be only *linear* and *without bias*.

  If fast_weights_eq_contraint == True:
    * We impose the equality constraint (r_j, s_j) = (u_j, v_j)

  If regularize_fast_weights == True, we have:
    * Assuming lambdas_ij and L2 coefficients h_ij
      (i in {1, ..., n} and j in {1, ..., ensemble_size}).
    * Denoting K_ij = (W * (r_j s_j^T)) + (e1(lambdas_ij) * (W' u_j v_j^T))

    1/(n*ensemble_size) sum_i,j h_ij || K_ij ||^2.

  Else (regularize_fast_weights == False) we have
    * Denoting Q_ij = W + (e1(lambdas_ij) * W')

    1/(n*ensemble_size) sum_i,j h_ij || Q_ij ||^2.

  """

  def __init__(self,
               units,
               lambda_key_to_index,
               rank=1,
               ensemble_size=4,
               activation=None,
               use_bias=True,
               alpha_initializer='ones',
               gamma_initializer='ones',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               regularize_fast_weights=True,
               fast_weights_eq_contraint=False,
               **kwargs):

    super().__init__(**kwargs)

    assert rank == 1, 'Self-tuned layers only support rank-1 fast weights.'
    assert_msg = 'Self-tuned layers handle their regularization seperately.'
    assert kernel_regularizer is None, assert_msg
    assert bias_regularizer is None, assert_msg

    self.lambda_key_to_index = lambda_key_to_index
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.activation = tf.python.keras.activations.get(activation)
    self.regularize_fast_weights = regularize_fast_weights
    self.fast_weights_eq_contraint = fast_weights_eq_contraint

    self.dense = _DenseBatchEnsembleNoFastWeights(
        units,
        rank=rank,
        ensemble_size=ensemble_size,
        activation=None,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

    self.delta_dense = _DenseBatchEnsembleNoFastWeights(
        units,
        rank=rank,
        ensemble_size=ensemble_size,
        activation=None,
        use_bias=False,  # bias of self-tuned part handled separately
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

    self.ensemble_size = self.dense.ensemble_size
    self.units = self.dense.units
    self.use_bias = use_bias
    self.bias_initializer = self.dense.bias_initializer

  def _add_weight(self, name, shape):
    assert ('alpha' in name) or ('gamma' in name)
    return self.add_weight(
        name=name,
        shape=shape,
        initializer=self.alpha_initializer
        if 'alpha' in name else self.gamma_initializer,
        trainable=True,
        dtype=self.dtype)

  def build(self, input_shape):

    # input_shape = [(None, data_dim), (None, lambdas_dim), (None, e_dim)]
    input_shape = tf.TensorShape(input_shape[0])
    input_dim = input_shape[-1]
    alpha_shape = [self.ensemble_size, input_dim]
    gamma_shape = [self.ensemble_size, self.units]

    self.dense.alpha = self._add_weight('alpha', alpha_shape)
    self.dense.gamma = self._add_weight('gamma', gamma_shape)

    if self.fast_weights_eq_contraint:
      self.delta_dense.alpha = self.dense.alpha
      self.delta_dense.gamma = self.dense.gamma
    else:
      # we follow the keras naming convention with '_1'
      self.delta_dense.alpha = self._add_weight('alpha_1', alpha_shape)
      self.delta_dense.gamma = self._add_weight('gamma_1', gamma_shape)

    if self.use_bias:
      self.bias = self.add_weight('bias',
                                  shape=(self.ensemble_size, self.units),
                                  initializer=self.bias_initializer,
                                  trainable=True,
                                  dtype=self.dtype)
    self.built = True

  def call(self, inputs):

    data, lambdas, e = inputs
    e1, e2 = e[:, :self.units], e[:, self.units:]

    output = self.dense(data)
    delta_kernel = self.delta_dense(data) * e1
    output += delta_kernel

    batch_size = tf.shape(data)[0]
    self.add_loss(self._get_mean_l2_regularizer(lambdas, e1, e2, batch_size))

    if self.use_bias:
      examples_per_model = batch_size // self.ensemble_size

      e2 = tf.reshape(e2, (self.ensemble_size, examples_per_model, self.units))
      delta_bias = tf.expand_dims(self.bias, 1) * e2
      delta_bias = tf.reshape(delta_bias, (batch_size, self.units))
      output += delta_bias

    if self.activation is not None:
      return self.activation(output)

    return output

  def _get_equivalent_kernels(self, kernel, alpha, gamma):
    """Compute equivalent kernels for all ensemble members."""
    k = tf.expand_dims(kernel, 0)  # (1, in_dim, units)
    if self.regularize_fast_weights:
      a = tf.expand_dims(alpha, -1)  # (ens_size, in_dim, 1)
      g = tf.expand_dims(gamma, 1)  # (ens_size, 1, units)
      kernels = k * a * g  # (ens_size, in_dim, units)
    else:
      kernels = tf.tile(k, [self.ensemble_size, 1, 1])
    return kernels

  def _get_mean_l2_regularizer_helper(self, w, u, e, l2):
    """Compute 1/n sum_i^n 1/k sum_j^k l2_{i,j} | w_j + u_j*e_{i,j} |_2^2."""

    # The arguments have the form:
    #   w in R^{k x a x b} with w_j in R^{a x b}
    #   u in R^{k x a x b} with u_j in R^{a x b}
    #   e in R^{k x n x b} with e_{i,j} in R^{1 x b}
    #   l2 in R^{k x n x 1}

    sq_w = tf.reduce_sum(tf.square(w), [1, 2], keepdims=True)  # (k, 1, 1)
    term1 = tf.reduce_mean(sq_w * l2)

    mean_e_l2 = tf.reduce_mean(e * l2, 1, keepdims=True)  # (k, 1, b)
    v = u * mean_e_l2
    wtv = tf.reduce_mean(tf.matmul(w, v, transpose_a=True), 0)  # (b, b)
    term2 = 2. * tf.linalg.trace(wtv)

    sq_u = tf.square(u)  # (k, a, b)
    mean_sq_e_l2 = tf.reduce_mean(
        tf.square(e) * l2, 1, keepdims=True)  # (k,1,b)
    term3 = tf.reduce_mean(tf.reduce_sum(sq_u * mean_sq_e_l2, [1, 2]))

    output = term1 + term2 + term3
    return output

  def _get_mean_l2_regularizer(self, lambdas, e1, e2, batch_size):

    # l2 regularization term for the kernel
    l2_k = get_lambda(
        lambdas,
        lambda_type='l2_kernel',
        layer_name=self.name,
        lambda_key_to_index=self.lambda_key_to_index)

    examples_per_model = batch_size // self.ensemble_size

    dense_kernel = self.dense.kernel
    kernels = self._get_equivalent_kernels(dense_kernel,
                                           self.dense.alpha,
                                           self.dense.gamma)

    delta_dense_kernel = self.delta_dense.kernel
    delta_kernels = self._get_equivalent_kernels(delta_dense_kernel,
                                                 self.delta_dense.alpha,
                                                 self.delta_dense.gamma)

    e1 = tf.reshape(e1, (self.ensemble_size, examples_per_model, self.units))
    l2_k = tf.reshape(l2_k, (self.ensemble_size, examples_per_model, 1))

    l2_regularizer = self._get_mean_l2_regularizer_helper(
        kernels, delta_kernels, e1, l2_k)

    if not self.use_bias:
      return l2_regularizer

    # l2 regularization term for the bias
    l2_bias = get_lambda(
        lambdas,
        lambda_type='l2_bias',
        layer_name=self.name,
        lambda_key_to_index=self.lambda_key_to_index)

    bias = tf.expand_dims(self.dense.ensemble_bias, 1)  # (ens_size, 1, units)
    delta_bias = tf.expand_dims(self.bias, 1)  # (ens_size, 1, units)
    e2 = tf.reshape(e2, (self.ensemble_size, examples_per_model, self.units))
    l2_bias = tf.reshape(l2_bias, (self.ensemble_size, examples_per_model, 1))

    l2_regularizer += self._get_mean_l2_regularizer_helper(
        bias, delta_bias, e2, l2_bias)

    return l2_regularizer

  def get_config(self):
    config = {
        'units':
            self.units,
        'lambda_key_to_index':
            self.lambda_key_to_index,
        'rank':
            self.dense.rank,
        'ensemble_size':
            self.ensemble_size,
        'activation':
            tf.python.keras.activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'alpha_initializer':
            initializers.serialize(self.alpha_initializer),
        'gamma_initializer':
            initializers.serialize(self.gamma_initializer),
        'kernel_initializer':
            initializers.serialize(self.dense.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.dense.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.dense.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.dense.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.dense.bias_constraint),
        'regularize_fast_weights':
            self.regularize_fast_weights,
        'fast_weights_eq_contraint':
            self.fast_weights_eq_contraint
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


def get_layer_name_identifier(layer_name):
  """Converts the layer name into a identifier to access lambda_key_to_index.

  As identifier the layer_name is used, but the part encapsulated by the
  character '/' is ignored. Useful if the Hyper-BatchEnsemble should use the
   same hyperparameter for a group of self tuned layers.

  Example:
    * layer_name='conv_2' returns 'conv_2'
    * layer_name='group_1/conv_3/' returns 'group_1'

  Args:
    layer_name: string.

  Returns:
    identifier: string, to be used to access lambda_key_to_index.
  """
  ignore_start = layer_name.find('/')
  ignore_end = layer_name.find('/', ignore_start + 1)
  if ignore_start > -1 and ignore_end > -1:
    layer_name = layer_name[:ignore_start] + layer_name[ignore_end + 1:]

  return layer_name


def get_lambda(lambdas, lambda_type, layer_name, lambda_key_to_index):
  """Extract the column in lambdas corresponding to the requested HP."""
  assert lambda_type in LAMBDA_TYPE

  identifier = get_layer_name_identifier(layer_name)
  index = lambda_key_to_index[identifier + '_' + lambda_type]
  return tf.reshape(lambdas[:, index], (-1, 1))


@utils.add_weight
class DenseRank1(tf.python.keras.layers.Dense):
  """A rank-1 Bayesian neural net dense layer (Dusenberry et al., 2020).

  The argument ensemble_size selects the number of mixture components over all
  weights, i.e., an ensemble of size `ensemble_size`. The layer performs a
  forward pass by enumeration, returning a forward pass under each mixture
  component. It takes an input tensor of shape
  [ensemble_size*examples_per_model, input_dim] and returns an output tensor of
  shape [ensemble_size*examples_per_model, units].

  To use a different batch for each mixture, take a minibatch of size
  ensemble_size*examples_per_model. To use the same batch for each mixture, get
  a minibatch of size examples_per_model and tile it by ensemble_size before
  applying any ensemble layers.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               alpha_initializer='trainable_normal',
               gamma_initializer='trainable_normal',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               alpha_regularizer='normal_kl_divergence',
               gamma_regularizer='normal_kl_divergence',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               alpha_constraint=None,
               gamma_constraint=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_additive_perturbation=False,
               min_perturbation_value=-10,
               max_perturbation_value=10,
               ensemble_size=1,
               **kwargs):
    super().__init__(
        units=units,
        use_bias=False,
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=None,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=None,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=None,
        **kwargs)
    self.units = units
    self.ensemble_activation = tf.python.keras.activations.get(activation)
    self.use_ensemble_bias = use_bias
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.ensemble_bias_initializer = initializers.get(bias_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.ensemble_bias_regularizer = regularizers.get(bias_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.ensemble_bias_constraint = constraints.get(bias_constraint)
    self.use_additive_perturbation = use_additive_perturbation
    self.min_perturbation_value = min_perturbation_value
    self.max_perturbation_value = max_perturbation_value
    self.ensemble_size = ensemble_size

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    super().build(input_shape)

    input_dim = input_shape[-1]
    self.alpha = self.add_weight(
        name='alpha',
        shape=[self.ensemble_size, input_dim],
        initializer=self.alpha_initializer,
        regularizer=self.alpha_regularizer,
        constraint=self.alpha_constraint,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        name='gamma',
        shape=[self.ensemble_size, self.units],
        initializer=self.gamma_initializer,
        regularizer=self.gamma_regularizer,
        constraint=self.gamma_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_ensemble_bias:
      self.ensemble_bias = self.add_weight(
          name='ensemble_bias',
          shape=[self.ensemble_size, self.units],
          initializer=self.ensemble_bias_initializer,
          regularizer=self.ensemble_bias_regularizer,
          constraint=self.ensemble_bias_constraint,
          trainable=True,
          dtype=self.dtype)
      self.ensemble_bias_shape = self.ensemble_bias.shape
    else:
      self.ensemble_bias = None
      self.ensemble_bias_shape = None
    self.alpha_shape = self.alpha.shape
    self.gamma_shape = self.gamma.shape
    self.built = True

  def call(self, inputs):
    batch_size = tf.shape(inputs)[0]
    input_dim = self.alpha_shape[-1]
    examples_per_model = batch_size // self.ensemble_size
    # NOTE: This restricts this layer from being called on tensors of ndim > 2.
    inputs = tf.reshape(
        inputs, [self.ensemble_size, examples_per_model, input_dim])

    # Sample parameters for each example.
    if isinstance(self.alpha_initializer, tf.python.keras.layers.Layer):
      alpha = tf.clip_by_value(
          self.alpha_initializer(
              self.alpha_shape,
              self.dtype).distribution.sample(examples_per_model),
          self.min_perturbation_value,
          self.max_perturbation_value)
      alpha = tf.transpose(alpha, [1, 0, 2])
    else:
      alpha = tf.expand_dims(self.alpha, 1)
    if isinstance(self.gamma_initializer, tf.python.keras.layers.Layer):
      gamma = tf.clip_by_value(
          self.gamma_initializer(
              self.gamma_shape,
              self.dtype).distribution.sample(examples_per_model),
          self.min_perturbation_value,
          self.max_perturbation_value)
      gamma = tf.transpose(gamma, [1, 0, 2])
    else:
      gamma = tf.expand_dims(self.gamma, 1)

    if self.use_additive_perturbation:
      outputs = super().call(inputs + alpha) + gamma
    else:
      outputs = super().call(inputs * alpha) * gamma

    if self.use_ensemble_bias:
      if isinstance(self.ensemble_bias_initializer, tf.python.keras.layers.Layer):
        bias = self.ensemble_bias_initializer(
            self.ensemble_bias_shape,
            self.dtype).distribution.sample(examples_per_model)
        bias = tf.transpose(bias, [1, 0, 2])
      else:
        bias = tf.expand_dims(self.ensemble_bias, 1)
      outputs += bias

    if self.ensemble_activation is not None:
      outputs = self.ensemble_activation(outputs)
    outputs = tf.reshape(outputs, [batch_size, self.units])
    return outputs

  def get_config(self):
    config = {
        'ensemble_activation':
            tf.python.keras.activations.serialize(self.ensemble_activation),
        'use_ensemble_bias':
            self.use_ensemble_bias,
        'alpha_initializer':
            initializers.serialize(self.alpha_initializer),
        'gamma_initializer':
            initializers.serialize(self.gamma_initializer),
        'ensemble_bias_initializer':
            initializers.serialize(self.ensemble_bias_initializer),
        'alpha_regularizer':
            regularizers.serialize(self.alpha_regularizer),
        'gamma_regularizer':
            regularizers.serialize(self.gamma_regularizer),
        'ensemble_bias_regularizer':
            regularizers.serialize(self.ensemble_bias_regularizer),
        'alpha_constraint':
            constraints.serialize(self.alpha_constraint),
        'gamma_constraint':
            constraints.serialize(self.gamma_constraint),
        'ensemble_bias_constraint':
            constraints.serialize(self.ensemble_bias_constraint),
        'ensemble_size':
            self.ensemble_size,
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


@utils.add_weight
class CondDense(tf.python.keras.layers.Dense):
  """Conditional dense layer.

  This layer extends the base dense layer to compute example-dependent
  parameters. A CondDense layer has 'num_experts` kernels and biases. It
  computes a kernel and bias for each example as a weighted sum of experts
  using the input example-dependent routing weights, then applies matrix
  multiplication to each example.

  Attributes:
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
  """

  def __init__(self,
               units,
               num_experts=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super().__init__(
        units=units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondDense layer must have at least one expert.')
    self.num_experts = num_experts

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.input_dim = int(input_shape[-1])
    self.kernel_shape = [self.input_dim, self.units]
    cond_dense_kernel_shape = (self.num_experts, self.input_dim * self.units)
    self.cond_dense_kernel = self.add_weight(
        name='cond_dense_kernel',
        shape=cond_dense_kernel_shape,
        initializer=initializers.get_condconv_initializer(
            self.kernel_initializer, self.num_experts, self.kernel_shape),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias_shape = (self.units,)
      cond_dense_bias_shape = (self.num_experts, self.units)
      self.cond_dense_bias = self.add_weight(
          name='cond_dense_bias',
          shape=cond_dense_bias_shape,
          initializer=initializers.get_condconv_initializer(
              self.bias_initializer, self.num_experts, self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent kernels
    inputs = tf.expand_dims(inputs, 1)  # shape = [batch_size, 1, input_dim]
    # routing_weights is of shape [batch_size, num_experts]
    # self.cond_dense_kernel is of shape [num_experts, input_dim * hidden_dim]
    kernels = tf.linalg.matmul(routing_weights, self.cond_dense_kernel)
    # kernels is of shape [batch_size, input_dim, hidden_dim]
    kernels = tf.reshape(kernels,
                         [-1, self.input_dim, self.units])
    outputs = tf.linalg.matmul(inputs, kernels)  # [batch, 1, hidden_dim]
    outputs = tf.reshape(outputs, [-1, self.units])  # [batch, hidden_dim]
    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.linalg.matmul(routing_weights, self.cond_dense_bias)
      outputs += biases
    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {'num_experts': self.num_experts}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
