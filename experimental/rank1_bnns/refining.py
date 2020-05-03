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

# Lint as: python3
"""Utilities for sampling"""
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from experimental.rank1_bnns import rank1_bnn_layers


def get_auxiliary_posterior(posterior_mean,
                            posterior_scale,
                            prior_mean,
                            prior_scale,
                            auxiliary_scale):
  """Calculates the posterior of an additive Gaussian auxiliary variable.
   q(a)=\int p(a|w)q(w)dw.
  """
  prior_var = tf.math.pow(prior_scale, 2)
  posterior_var = tf.math.pow(posterior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)
  aux_div_prior_var = auxiliary_var / prior_var
  auxiliary_posterior_mean = (posterior_mean - prior_mean) * aux_div_prior_var
  auxiliary_posterior_var = (posterior_var * tf.math.pow(auxiliary_var, 2)
                             / tf.math.pow(prior_var, 2)
                             + aux_div_prior_var * (prior_var - auxiliary_var))
  return auxiliary_posterior_mean, tf.sqrt(auxiliary_posterior_var)


def get_conditional_prior(prior_mean,
                          prior_scale,
                          auxiliary_scale,
                          auxiliary_sample):
  """Calculates the conditional prior given an auxiliary variable.
  p(w|a).
  """
  prior_var = tf.math.pow(prior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)
  return prior_mean + auxiliary_sample, tf.sqrt(prior_var - auxiliary_var)


def get_conditional_posterior(posterior_mean,
                              posterior_scale,
                              prior_mean,
                              prior_scale,
                              auxiliary_scale,
                              auxiliary_sample):
  """Calculates the conditional posterior given an additive auxiliary variable.
  q(w|a)\propto p(a|w)q(w).
  """
  prior_var = tf.math.pow(prior_scale, 2)
  posterior_var = tf.math.pow(posterior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)\

  cond_x_prior_var = (prior_var - auxiliary_var) * prior_var
  aux_x_post_var = auxiliary_var * posterior_var
  denom = cond_x_prior_var + aux_x_post_var
  conditional_mean = (prior_mean + (auxiliary_sample * posterior_var *
                                    prior_var + (posterior_mean - prior_mean)
                                    * cond_x_prior_var) / denom)
  conditional_var = posterior_var * cond_x_prior_var / denom
  return conditional_mean, tf.sqrt(conditional_var)


def sample_rank1_auxiliaries(model, auxiliary_var_ratio):
  """Samples additive Gaussian auxiliary variables for the layer.
  For every rank1 BNN layer, then it samples additive Gaussian auxiliary
  variables for alpha and gamma. It is assumed that the priors and posteriors
  of alpha and gamma are both Gaussians.

  Args:
      model: Keras model.
      auxiliary_var_ratio: The ratio of the variance of the auxiliary variable
      to the variance of the prior. (0 < auxiliary_var_ratio < 1)
  """
  for layer in model.layers:
    if (isinstance(layer, rank1_bnn_layers.DenseRank1) or
        isinstance(layer, rank1_bnn_layers.Conv2DRank1)):
      for initializer, regularizer in [(layer.alpha_initializer,
                                        layer.alpha_regularizer),
                                       (layer.gamma_initializer,
                                        layer.gamma_regularizer)]:
        posterior_mean = initializer.mean
        unconstrained_posterior_scale = initializer.stddev
        print(unconstrained_posterior_scale)
        posterior_scale = initializer.stddev_constraint(
          unconstrained_posterior_scale)
        prior_mean = regularizer.mean
        prior_scale = regularizer.stddev
        auxiliary_scale_ratio = np.sqrt(auxiliary_var_ratio)
        auxiliary_scale = tf.cast(auxiliary_scale_ratio * prior_scale,
                                  dtype=posterior_mean.dtype)
        a_mean, a_scale = get_auxiliary_posterior(posterior_mean,
                                                  posterior_scale,
                                                  prior_mean,
                                                  prior_scale,
                                                  auxiliary_scale)
        auxiliary_sample = tfp.distributions.Normal(loc=a_mean,
                                                    scale=a_scale).sample()
        new_posterior_mean, new_posterior_scale = get_conditional_posterior(
          posterior_mean,
          posterior_scale,
          prior_mean,
          prior_scale,
          auxiliary_scale,
          auxiliary_sample)
        new_prior_mean, new_prior_scale = get_conditional_prior(
          prior_mean,
          prior_scale,
          auxiliary_scale,
          auxiliary_sample)
        posterior_mean.assign(new_posterior_mean)
        unconstrained_posterior_scale.assign(
          tfp.math.softplus_inverse(new_posterior_scale))
        regularizer.mean = new_prior_mean.numpy()
        regularizer.stddev = new_prior_scale.numpy()
