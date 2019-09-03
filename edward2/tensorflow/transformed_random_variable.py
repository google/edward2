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

"""Reversible layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2 import trace
from edward2.tensorflow import random_variable
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


class TransformedDistribution(tfp.distributions.Distribution):
  """Distribution of f(x), where x ~ p(x) and f is reversible."""

  def __init__(self, base, reversible_layer, name=None):
    """Constructs a transformed distribution.

    Args:
      base: Base distribution.
      reversible_layer: Callable with methods `reverse` and `log_det_jacobian`.
      name: Name for scoping operations in the class.
    """
    self.base = base
    self.reversible_layer = reversible_layer
    if name is None:
      name = reversible_layer.name + base.name
    super(TransformedDistribution, self).__init__(
        base.dtype,
        base.reparameterization_type,
        base.validate_args,
        base.allow_nan_stats,
        parameters=dict(locals()),
        name=name)

  def _event_shape_tensor(self):
    return self.base.event_shape_tensor()

  def _event_shape(self):
    return self.base.event_shape

  def _batch_shape_tensor(self):
    return self.base.batch_shape_tensor()

  def _batch_shape(self):
    return self.base.batch_shape

  def __getitem__(self, slices):
    overrides = {'base': self.base[slices]}
    return self.copy(**overrides)

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    x = self.base.sample(sample_shape, seed, **kwargs)
    y = self.reversible_layer(x)
    return y

  def _log_prob(self, value):
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.log_prob(x) + log_det_jacobian

  def _prob(self, value):
    if not hasattr(self.base, '_prob'):
      return tf.exp(self.log_prob(value))
    x = self.reversible_layer.reverse(value)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(value)
    return self.base.prob(x) * tf.exp(log_det_jacobian)

  def _log_cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_cdf(x)

  def _cdf(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.cdf(x)

  def _log_survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.log_survival_function(x)

  def _survival_function(self, value):
    x = self.reversible_layer.reverse(value)
    return self.base.survival_function(x)

  def _quantile(self, value):
    inverse_cdf = self.base.quantile(value)
    return self.reversible_layer(inverse_cdf)

  def _entropy(self):
    dummy = tf.zeros(
        tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], 0),
        dtype=self.dtype)
    log_det_jacobian = self.reversible_layer.log_det_jacobian(dummy)
    entropy = self.base.entropy() - log_det_jacobian
    return entropy


@trace.traceable
def TransformedRandomVariable(rv,  # pylint: disable=invalid-name
                              reversible_layer,
                              name=None,
                              sample_shape=(),
                              value=None):
  """Random variable for f(x), where x ~ p(x) and f is reversible."""
  return random_variable.RandomVariable(
      distribution=TransformedDistribution(rv.distribution,
                                           reversible_layer,
                                           name=name),
      sample_shape=sample_shape,
      value=value)
