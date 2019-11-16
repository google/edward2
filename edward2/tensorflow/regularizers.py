# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

This module extends `tf.keras.regularizers` with two features:

1. Regularizers which compute using any weight random variables' distribution.
For example, consider a regularizer which computes an analytic KL
divergence given an input ed.Normal weight.
2. "Trainable regularizers" are regularizers which may themselves carry
parameters. For example, consider a weight regularizer which computes a
KL divergence from the weights towards a learnable prior.

One subtlety is how `tf.keras.constraints` are used on the parameters of
trainable regularizers. Typically, Keras constraints are used with projected
gradient descent, where one performs unconstrained optimization and then applies
a projection (the constraint) after each gradient update. To stay in line with
probabilistic literature, trainable regularizers apply constraints on the
`tf.Variables` themselves (i.e., a constrained parameterization) and do not
apply projections during optimization.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import random_variable
import six
import tensorflow.compat.v2 as tf


class HalfCauchyKLDivergence(tf.keras.regularizers.Regularizer):
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


class LogUniformKLDivergence(tf.keras.regularizers.Regularizer):
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
                                           tf.keras.backend.epsilon())
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


class NormalKLDivergence(tf.keras.regularizers.Regularizer):
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


class TrainableNormalKLDivergenceStdDev(tf.keras.layers.Layer):
  """Normal KL divergence with trainable stddev parameter."""

  def __init__(self,
               mean=0.,
               stddev_initializer=tf.keras.initializers.TruncatedNormal(
                   mean=0.5413248, stddev=0.1),  # mean=softplus_inverse(1.)
               stddev_regularizer=None,
               stddev_constraint='softplus',
               scale_factor=1.,
               seed=None,
               **kwargs):
    super(TrainableNormalKLDivergenceStdDev, self).__init__(**kwargs)
    self.mean = mean
    self.stddev_initializer = tf.keras.initializers.get(stddev_initializer)
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
            tf.keras.initializers.serialize(self.stddev_initializer),
        'stddev_regularizer': serialize(self.stddev_regularizer),
        'stddev_constraint': constraints.serialize(self.stddev_constraint),
        'scale_factor': self.scale_factor,
        'seed': self.seed,
    }


# Compatibility aliases, following tf.keras

# pylint: disable=invalid-name
half_cauchy_kl_divergence = HalfCauchyKLDivergence
log_uniform_kl_divergence = LogUniformKLDivergence
normal_kl_divergence = NormalKLDivergence
trainable_normal_kl_divergence_stddev = TrainableNormalKLDivergenceStdDev
# pylint: enable=invalid-name

# Utility functions, following tf.keras


def serialize(initializer):
  return tf.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizers')


def get(identifier, value=None):
  """Getter for loading from strings; falls back to Keras as needed."""
  if value is None:
    value = identifier
  if identifier is None:
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      pass
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      pass
  elif callable(identifier):
    return identifier
  return tf.keras.regularizers.get(value)
