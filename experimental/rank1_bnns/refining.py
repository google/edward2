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

from experimental.rank1_bnns.rank1_bnn_layers import DenseRank1, Conv2DRank1


def get_auxiliary_posterior(posterior_mean,
                            posterior_scale,
                            prior_mean,
                            prior_scale,
                            auxiliary_scale):
  """ Calculate the posterior distribution of an additive Gaussian """
  """ auxiliary variable. q(a)=\int p(a|w)q(w)dw. """
  prior_var = tf.math.pow(prior_scale, 2)
  posterior_var = tf.math.pow(posterior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)
  auxiliary_posterior_mean = (posterior_mean - prior_mean) \
      * auxiliary_var / prior_var
  auxiliary_posterior_var = posterior_var * tf.math.pow(auxiliary_var, 2) \
      / tf.math.pow(prior_var, 2) + auxiliary_var * \
      (prior_var - auxiliary_var) / prior_var
  return auxiliary_posterior_mean, tf.sqrt(auxiliary_posterior_var)


def get_conditional_prior(prior_mean,
                          prior_scale,
                          auxiliary_scale,
                          auxiliary_sample):
  """ Calculate the conditional prior given the value of an additive """
  """ Gaussian auxiliary variable. p(w|a). """
  prior_var = tf.math.pow(prior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)
  return prior_mean + auxiliary_sample, tf.sqrt(prior_var - auxiliary_var)


def get_conditional_posterior(posterior_mean,
                              posterior_scale,
                              prior_mean,
                              prior_scale,
                              auxiliary_scale,
                              auxiliary_sample):
  """ Calculate the conditional posterior given the value of an additive """
  """ Gaussian auxiliary variable. q(w|a)\propto p(a|w)q(w). """
  prior_var = tf.math.pow(prior_scale, 2)
  posterior_var = tf.math.pow(posterior_scale, 2)
  auxiliary_var = tf.math.pow(auxiliary_scale, 2)
  conditional_mean = (prior_mean + (auxiliary_sample * posterior_var *
                                    prior_var + (posterior_mean - prior_mean)
                                    * (prior_var - auxiliary_var)
                                    * prior_var)
                      / (posterior_var * auxiliary_var + prior_var *
                         (prior_var - auxiliary_var)))
  conditional_var = posterior_var * prior_var * (prior_var - auxiliary_var) / \
                    (auxiliary_var * posterior_var + prior_var *
                     (prior_var - auxiliary_var))
  return conditional_mean, tf.sqrt(conditional_var)


def sample_rank1_auxiliaries(model, auxiliary_var_ratio):
  for layer in model.layers:
    if isinstance(layer, DenseRank1) or isinstance(layer, Conv2DRank1):
      for rv_name, rv, regularizer in [('alpha', layer.alpha,
                                        layer.alpha_regularizer),
                                       ('gamma', layer.gamma,
                                        layer.gamma_regularizer)]:
        posterior_mean = rv.distribution.distribution.loc
        posterior_scale = rv.distribution.distribution.scale
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
        for v in layer.variables:
          if rv_name + '/mean' in v.name:
            posterior_mean_variable = v
            posterior_mean_variable.assign(new_posterior_mean)
          if rv_name + '/stddev' in v.name:
            posterior_scale_untransformed = v
            posterior_scale_untransformed.assign(
                tfp.math.softplus_inverse(new_posterior_scale))
          regularizer.mean = new_prior_mean.numpy()
          regularizer.stddev = new_prior_scale.numpy()
