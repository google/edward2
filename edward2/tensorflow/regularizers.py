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

"""Regularizers.

This module extends `tf.python.keras.regularizers` with two features:

1. Regularizers which compute using any weight random variables' distribution.
For example, consider a regularizer which computes an analytic KL
divergence given an input ed.Normal weight.
2. "Trainable regularizers" are regularizers which may themselves carry
parameters. For example, consider a weight regularizer which computes a
KL divergence from the weights towards a learnable prior.

One subtlety is how `tf.python.keras.constraints` are used on the parameters of
trainable regularizers. Typically, Keras constraints are used with projected
gradient descent, where one performs unconstrained optimization and then applies
a projection (the constraint) after each gradient update. To stay in line with
probabilistic literature, trainable regularizers apply constraints on the
`tf.Variables` themselves (i.e., a constrained parameterization) and do not
apply projections during optimization.
"""

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import random_variable
import tensorflow as tf


class CauchyKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the Cauchy distribution."""

  def __init__(self, loc=0., scale=1., scale_factor=1.):
    """Constructs regularizer where default uses the standard Cauchy."""
    self.loc = loc
    self.scale = scale
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization using an unbiased Monte Carlo estimate."""
    prior = generated_random_variables.Independent(
        generated_random_variables.Cauchy(
            loc=tf.broadcast_to(self.loc, x.distribution.event_shape),
            scale=tf.broadcast_to(self.scale, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    negative_entropy = x.distribution.log_prob(x)
    cross_entropy = -prior.distribution.log_prob(x)
    return self.scale_factor * (negative_entropy + cross_entropy)

  def get_config(self):
    return {
        'loc': self.loc,
        'scale': self.scale,
        'scale_factor': self.scale_factor,
    }


class HalfCauchyKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the half-Cauchy distribution."""

  def __init__(self, loc=0., scale=1., scale_factor=1.):
    """Constructs regularizer where default uses the standard half-Cauchy."""
    self.loc = loc
    self.scale = scale
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization using an unbiased Monte Carlo estimate."""
    prior = generated_random_variables.Independent(
        generated_random_variables.HalfCauchy(
            loc=tf.broadcast_to(self.loc, x.distribution.event_shape),
            scale=tf.broadcast_to(self.scale, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    negative_entropy = x.distribution.log_prob(x)
    cross_entropy = -prior.distribution.log_prob(x)
    return self.scale_factor * (negative_entropy + cross_entropy)

  def get_config(self):
    return {
        'loc': self.loc,
        'scale': self.scale,
        'scale_factor': self.scale_factor,
    }


class LogUniformKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the log-uniform distribution."""

  def __init__(self, scale_factor=1.):
    """Constructs regularizer."""
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an ed.Normal random variable as input."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable (for correct math, '
                       'an ed.Normal random variable).')
    # Clip magnitude of dropout rate, where we get the dropout rate alpha from
    # the additive parameterization (Molchanov et al., 2017): for weight ~
    # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
    mean = x.distribution.mean()
    log_variance = tf.math.log(x.distribution.variance())
    log_alpha = log_variance - tf.math.log(tf.square(mean) +
                                           tf.python.keras.backend.epsilon())
    log_alpha = tf.clip_by_value(log_alpha, -8., 8.)

    # Set magic numbers for cubic polynomial approx. (Molchanov et al., 2017).
    k1 = 0.63576
    k2 = 1.8732
    k3 = 1.48695
    c = -k1
    output = tf.reduce_sum(k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) +
                           -0.5 * tf.math.log1p(tf.exp(-log_alpha)) + c)
    return self.scale_factor * output

  def get_config(self):
    return {
        'scale_factor': self.scale_factor,
    }


class LogNormalKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the log normal distribution."""

  def __init__(self, loc=0., scale=1., scale_factor=1.):
    """Constructs regularizer where default is a KL towards a std log normal."""
    self.loc = loc
    self.scale = scale
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an input ed.RandomVariable."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    prior = generated_random_variables.Independent(
        generated_random_variables.LogNormal(
            loc=tf.broadcast_to(self.loc, x.distribution.event_shape),
            scale=tf.broadcast_to(self.scale, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    regularization = x.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'loc': self.loc,
        'scale': self.scale,
        'scale_factor': self.scale_factor,
    }


class NormalKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the normal distribution."""

  def __init__(self, mean=0., stddev=1., scale_factor=1.):
    """Constructs regularizer where default is a KL towards the std normal."""
    self.mean = mean
    self.stddev = stddev
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an input ed.RandomVariable."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    prior = generated_random_variables.Independent(
        generated_random_variables.Normal(
            loc=tf.broadcast_to(self.mean, x.distribution.event_shape),
            scale=tf.broadcast_to(self.stddev, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    regularization = x.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'mean': self.mean,
        'stddev': self.stddev,
        'scale_factor': self.scale_factor,
    }


class NormalEmpiricalBayesKLDivergence(NormalKLDivergence):
  """Normal prior with distribution on variance and using empirical Bayes.

  This regularizer uses a hierachical prior that shares a variance distribution
  across all weight elements (Wu et al., 2018):

  ```
  p(variance) = InverseGamma(variance | variance_concentration, variance_scale)
  p(weight[i, j]) = Normal(weight[i, j] | mean, variance),
  ```

  where `variance_concentration`, `variance_scale`, and `mean` are fixed. Given
  an input random variable q(weight), the regularizer computes a KL divergence
  towards the prior distribution p(weight, variance). The variance is fixed at
  the value variance*:

  ```
  R(weight)
  = KL( q(weight) deterministic(variance | variance*) || p(weight, scale) )
  = E [ log q(weight) + log deterministic(variance | variance*) -
        log p(weight, scale) ]
  = E [ log q(weight) - log p(weight | variance*) ] - log p(variance*)
  = KL( q(weight) || p(weight | variance*) ) - log p(variance*).
  ```

  We use Wu et al. (2018)'s closed-form solution for variance*. The estimate is
  approximate if the input random variable is not normally distributed.
  """

  def __init__(self,
               mean=0.,
               variance_concentration=2.01,
               variance_scale=0.101,
               scale_factor=1.):
    """Constructs regularizer."""
    self.variance_concentration = variance_concentration
    self.variance_scale = variance_scale
    super(NormalEmpiricalBayesKLDivergence, self).__init__(
        mean=mean,
        stddev=None,  # to be estimated at each call
        scale_factor=scale_factor)

  def __call__(self, x):
    """Computes regularization given an input ed.RandomVariable."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    # variance = (tr( sigma_q + mu_q mu_q^T ) + 2*beta) / (omega + 2*alpha + 2)
    trace_covariance = tf.reduce_sum(x.distribution.variance())
    trace_mean_outer_product = tf.reduce_sum(x.distribution.mean()**2)
    num_weights = tf.cast(tf.reduce_prod(x.shape), x.dtype)
    variance = ((trace_covariance + trace_mean_outer_product) +
                2. * self.variance_scale)
    variance /= num_weights + 2. * self.variance_concentration + 2.
    self.stddev = tf.sqrt(variance)

    variance_prior = generated_random_variables.InverseGamma(
        self.variance_concentration, self.variance_scale)
    regularization = super(NormalEmpiricalBayesKLDivergence, self).__call__(x)
    regularization -= (self.scale_factor *
                       variance_prior.distribution.log_prob(variance))
    return regularization

  def get_config(self):
    return {
        'mean': self.mean,
        'variance_concentration': self.variance_concentration,
        'variance_scale': self.variance_scale,
        'scale_factor': self.scale_factor,
    }


class NormalKLDivergenceWithTiedMean(tf.python.keras.regularizers.Regularizer):
  """KL with normal prior whose mean is fixed at the variational posterior's."""

  def __init__(self, stddev=1., scale_factor=1.):
    """Constructs regularizer."""
    self.stddev = stddev
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an ed.Normal random variable as input."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    prior = generated_random_variables.Independent(
        generated_random_variables.Normal(loc=x.distribution.mean(),
                                          scale=self.stddev).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    regularization = x.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'stddev': self.stddev,
        'scale_factor': self.scale_factor,
    }


class TrainableNormalKLDivergenceStdDev(tf.python.keras.layers.Layer):
  """Normal KL divergence with trainable stddev parameter."""

  def __init__(self,
               mean=0.,
               stddev_initializer=tf.python.keras.initializers.TruncatedNormal(
                   mean=0.5413248, stddev=0.1),  # mean=softplus_inverse(1.)
               stddev_regularizer=None,
               stddev_constraint='softplus',
               scale_factor=1.,
               seed=None,
               **kwargs):
    super(TrainableNormalKLDivergenceStdDev, self).__init__(**kwargs)
    self.mean = mean
    self.stddev_initializer = tf.python.keras.initializers.get(stddev_initializer)
    self.stddev_regularizer = get(stddev_regularizer)
    self.stddev_constraint = constraints.get(stddev_constraint)
    self.scale_factor = scale_factor

  def build(self, input_shape):
    self.stddev = self.add_weight(
        'stddev',
        shape=input_shape,
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=None,
        dtype=self.dtype,
        trainable=True)
    self.built = True

  def call(self, inputs):
    """Computes regularization given an input ed.RandomVariable."""
    if not isinstance(inputs, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    stddev = self.stddev
    if self.stddev_constraint:
      stddev = self.stddev_constraint(stddev)
    prior = generated_random_variables.Independent(
        generated_random_variables.Normal(
            loc=self.mean, scale=stddev).distribution,
        reinterpreted_batch_ndims=len(inputs.distribution.event_shape))
    regularization = inputs.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'loc': self.loc,
        'stddev_initializer':
            tf.python.keras.initializers.serialize(self.stddev_initializer),
        'stddev_regularizer': serialize(self.stddev_regularizer),
        'stddev_constraint': constraints.serialize(self.stddev_constraint),
        'scale_factor': self.scale_factor,
        'seed': self.seed,
    }


class UniformKLDivergence(tf.python.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to a uniform distribution.

  This regularizer computes the negative entropy of the input variable, which
  yields a value that is proportional to the KL divergence up to an additive
  constant. Assumes a uniform distribution over the support of the input random
  variable.
  """

  def __init__(self, scale_factor=1.):
    """Constructs regularizer."""
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an input ed.RandomVariable."""
    if not isinstance(x, random_variable.RandomVariable):
      raise ValueError('Input must be an ed.RandomVariable.')
    return self.scale_factor * -x.distribution.entropy()

  def get_config(self):
    return {
        'scale_factor': self.scale_factor,
    }


# Compatibility aliases, following tf.python.keras

# pylint: disable=invalid-name
cauchy_kl_divergence = CauchyKLDivergence
half_cauchy_kl_divergence = HalfCauchyKLDivergence
log_normal_kl_divergence = LogNormalKLDivergence
log_uniform_kl_divergence = LogUniformKLDivergence
normal_kl_divergence = NormalKLDivergence
normal_empirical_bayes_kl_divergence = NormalEmpiricalBayesKLDivergence
trainable_normal_kl_divergence_stddev = TrainableNormalKLDivergenceStdDev
uniform_kl_divergence = UniformKLDivergence
# pylint: enable=invalid-name

# Utility functions, following tf.python.keras


def serialize(initializer):
  return tf.python.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.python.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizers')


def get(identifier, value=None):
  """Getter for loading from strings; falls back to Keras as needed."""
  if value is None:
    value = identifier
  if identifier in (None, ''):
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      pass
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      pass
  elif callable(identifier):
    return identifier
  return tf.python.keras.regularizers.get(value)
