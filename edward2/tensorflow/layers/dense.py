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

"""Uncertainty-based dense layers."""

import math
from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import initializers
from edward2.tensorflow import random_variable
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


@utils.add_weight
class DenseReparameterization(tf.keras.layers.Dense):
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
    super(DenseReparameterization, self).__init__(
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
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super(DenseReparameterization, self).call(*args, **kwargs)


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
  model = tf.keras.Sequential([
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
      return super(DenseDVI, self).call(inputs)
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

    if self.activation in (tf.keras.activations.relu, tf.nn.relu):
      # Compute activation's moments with variable names from Wu et al. (2018).
      variance = tf.linalg.diag_part(covariance)
      scale = tf.sqrt(variance)
      mu = mean / (scale + tf.keras.backend.epsilon())
      mean = scale * soft_relu(mu)

      pairwise_variances = (tf.expand_dims(variance, -1) *
                            tf.expand_dims(variance, -2))  # [..., units, units]
      rho = covariance / tf.sqrt(pairwise_variances +
                                 tf.keras.backend.epsilon())
      rho = tf.clip_by_value(rho,
                             -1. / (1. + tf.keras.backend.epsilon()),
                             1. / (1. + tf.keras.backend.epsilon()))
      s = covariance / (rho + tf.keras.backend.epsilon())
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
      safe_gr = tf.abs(gr) + 0.5 * tf.keras.backend.epsilon()
      safe_rho = tf.abs(rho) + tf.keras.backend.epsilon()
      exp_negative_q = gr / (2. * math.pi) * tf.exp(
          -safe_rho / (2. * safe_gr * (1 + bar_rho)) +
          (gh - rho) / (safe_gr * safe_rho) * mu1 * mu2)
      covariance = s * (a + exp_negative_q)
    elif self.activation not in (tf.keras.activations.linear, None):
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
      return super(DenseFlipout, self).call(inputs)
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
    super(DenseVariationalDropout, self).__init__(
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
      return super(DenseVariationalDropout, self).call(inputs)
    self.call_weights()
    if training is None:
      training = tf.keras.backend.learning_phase()

    def dropped_inputs():
      """Forward pass with dropout."""
      # Clip magnitude of dropout rate, where we get the dropout rate alpha from
      # the additive parameterization (Molchanov et al., 2017): for weight ~
      # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
      mean = self.kernel.distribution.mean()
      log_variance = tf.math.log(self.kernel.distribution.variance())
      log_alpha = log_variance - tf.math.log(tf.square(mean) +
                                             tf.keras.backend.epsilon())
      log_alpha = tf.clip_by_value(log_alpha, -8., 8.)
      log_variance = log_alpha + tf.math.log(tf.square(mean) +
                                             tf.keras.backend.epsilon())

      if inputs.shape.ndims <= 2:
        means = tf.matmul(inputs, mean)
        stddevs = tf.sqrt(
            tf.matmul(tf.square(inputs), tf.exp(log_variance)) +
            tf.keras.backend.epsilon())
      else:
        means = tf.tensordot(inputs, mean, [[-1], [0]])
        stddevs = tf.sqrt(
            tf.tensordot(tf.square(inputs), tf.exp(log_variance), [[-1], [0]]) +
            tf.keras.backend.epsilon())
      if self.use_bias:
        means = tf.nn.bias_add(means, self.bias)
      outputs = generated_random_variables.Normal(loc=means, scale=stddevs)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True.
    training_value = utils.smart_constant_value(training)
    if training_value is not None:
      if training_value:
        return dropped_inputs()
      else:
        return super(DenseVariationalDropout, self).call(inputs)
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
    super(DenseHierarchical, self).__init__(
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
    super(DenseHierarchical, self).build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.local_scale_initializer, tf.keras.layers.Layer):
      self.local_scale = self.local_scale_initializer(self.local_scale.shape,
                                                      self.dtype)
    if isinstance(self.global_scale_initializer, tf.keras.layers.Layer):
      self.global_scale = self.global_scale_initializer(self.global_scale.shape,
                                                        self.dtype)
    super(DenseHierarchical, self).call_weights()

  def call(self, inputs, training=None):
    self.call_weights()
    # TODO(trandustin): Figure out what to set local/global scales to at test
    # time. Means don't exist for Half-Cauchy approximate posteriors.
    inputs *= self.local_scale[tf.newaxis, :] * self.global_scale
    return super(DenseHierarchical, self).call(inputs, training=training)


class DenseBatchEnsemble(tf.keras.layers.Layer):
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
    super(DenseBatchEnsemble, self).__init__(**kwargs)
    self.rank = rank
    self.ensemble_size = ensemble_size
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.dense = tf.keras.layers.Dense(
        units=units,
        use_bias=False,
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=None,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=None,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=None)
    self.units = self.dense.units

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
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
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=[self.ensemble_size, self.units],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
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
      outputs = self.dense(perturb_inputs)
      outputs = tf.reduce_sum(outputs * gamma, axis=0)
      outputs = tf.reshape(
          outputs, [self.ensemble_size, examples_per_model, -1])
    else:
      inputs = tf.reshape(
          inputs, [self.ensemble_size, examples_per_model, input_dim])
      alpha = tf.expand_dims(self.alpha, 1)
      gamma = tf.expand_dims(self.gamma, 1)
      outputs = self.dense(inputs * alpha) * gamma

    if self.use_bias:
      bias = tf.expand_dims(self.bias, 1)
      outputs += bias

    if self.activation is not None:
      outputs = self.activation(outputs)
    outputs = tf.reshape(outputs, [batch_size, self.units])
    return outputs

  def compute_output_shape(self, input_shape):
    return self.dense.compute_output_shape(input_shape)

  def get_config(self):
    config = {
        'ensemble_size': self.ensemble_size,
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'bias_constraint': constraints.serialize(self.bias_constraint),
    }
    new_config = super(DenseBatchEnsemble, self).get_config()
    new_config.update(self.dense.get_config())
    new_config.update(config)
    return new_config
