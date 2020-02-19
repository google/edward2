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

"""Build a simple, feed forward Bayesian neural network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.experimental.auxiliary_sampling.sampling import mean_field_fn

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def simple_net(n_examples, input_shape, output_scaler=1.):
  """Build a simple, feed forward Bayesian neural net."""
  p_fn, q_fn = mean_field_fn(empirical_bayes=True)

  def output_dist_fn(t):
    loc, untransformed_scale = t
    return tfd.Normal(loc=loc, scale=tf.nn.softplus(untransformed_scale))

  def normalized_kl_fn(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.to_float(n_examples)

  inputs = tf.keras.layers.Input(shape=input_shape)
  hidden = tfp.layers.DenseLocalReparameterization(
      50,
      activation='relu',
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          inputs)
  output = tfp.layers.DenseLocalReparameterization(
      1,
      activation='linear',
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          hidden)
  output_untransformed_scale = tfp.layers.VariableLayer(
      shape=(), initializer=tf.keras.initializers.Constant(-3.))(
          output)
  output_dist = tfp.layers.DistributionLambda(output_dist_fn)(
      (output * output_scaler, output_untransformed_scale))
  return tf.keras.models.Model(inputs=inputs, outputs=output_dist)
