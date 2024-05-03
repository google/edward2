# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Utilities for the auxiliary sampling algorithm."""

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

tfd = tfp.distributions

flags.DEFINE_float("mean_field_init_untransformed_scale", -7,
                   "Initial scale (before softplus) for mean field.")
FLAGS = flags.FLAGS


def mean_field_fn(empirical_bayes=False,
                  initializer=tf1.initializers.he_normal()):
  """Constructors for Gaussian prior and posterior distributions.

  Args:
    empirical_bayes (bool): Whether to train the variance of the prior or not.
    initializer (tf1.initializer): Initializer for the posterior means.
  Returns:
    prior, posterior (tfp.distribution): prior and posterior
    to be fed into a Bayesian Layer.
  """

  def prior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the prior distribution (tfp.distributions.Independent)."""
    softplus_inverse_scale = np.log(np.exp(1.) - 1.)

    istrainable = add_variable_fn(
        name=name + "_istrainable",
        shape=(),
        initializer=tf1.constant_initializer(1.),
        dtype=dtype,
        trainable=False)

    untransformed_scale = add_variable_fn(
        name=name + "_untransformed_scale",
        shape=(),
        initializer=tf1.constant_initializer(softplus_inverse_scale),
        dtype=dtype,
        trainable=empirical_bayes and trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale * istrainable + (1. - istrainable) *
                       tf1.stop_gradient(untransformed_scale)))
    loc = add_variable_fn(
        name=name + "_loc",
        shape=shape,
        initializer=tf1.constant_initializer(0.),
        dtype=dtype,
        trainable=False)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.istrainable = istrainable
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  def posterior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the posterior distribution (tfp.distributions.Independent)."""
    untransformed_scale = add_variable_fn(
        name=name + "_untransformed_scale",
        shape=shape,
        initializer=tf1.compat.v1.initializers.random_normal(
            mean=FLAGS.mean_field_init_untransformed_scale, stddev=0.1),
        dtype=dtype,
        trainable=trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale))
    loc = add_variable_fn(
        name=name + "_loc",
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        trainable=trainable)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  return prior, posterior


def sample_auxiliary_op(prior, posterior, aux_variance_ratio):
  r"""Sample the auxiliary variable and calculate the conditionals.

  Given a gaussian prior $$\mathcal{N}(\mu_z, \sigma^2_z)$$
  Define auxiliary variables $$z=a_1+a_2$$ with $$a_1=\mathcal{N}(0,
  \sigma_{a_1}^2)$$ and $$a_2=\mathcal{N}(\mu_z, \sigma_{a_2}^2)$$ with
  $$\frac{\sigma_{a_1}^2}{\sigma^2_z}=$$aux_variance_ratio and
  $$\sigma_{a_1}^2+\sigma_{a_2}^2=\sigma_z^2$$.
  From this, we can calculate the posterior of a1 and the conditional of z.

  Conditional:
  $$p(a_1|z) =  \mathcal{N}(z \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_z^2})$$

  Posterior of $$a_1$$:
  $$q(a_1) =\mathcal{N}(\mu_{q(z)} \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{q(z)}^2\sigma_{a_1}^4}{\sigma_{z}^4} +
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_{z}^2})$$

  Conditional posterior:
  $$q(z|a_1)=\frac{q(a_1|z)q(z)}{q(a_1)}$$

  $$q(z|a_1)=\mathcal{N}(\frac{a_1\sigma^2_{q(z)}\sigma^2_{z} +
  \mu_{q(z)}\sigma^2_{a_2}\sigma^2_{z}}{\sigma^2_{q(z)}\sigma^2_{a_1} +
  \sigma^2_z\sigma^2_{a_2}},
  \frac{\sigma^2_{q(z)}\sigma^2_z\sigma^2_{a_2}}{\sigma^2_{a_1}\sigma^2_{q(z)} +
  \sigma^2_{z}\sigma^2_{a_2}})$$.

  Args:
    prior: The prior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    posterior: The posterior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    aux_variance_ratio: Ratio of the variance of the auxiliary variable and the
      prior. The mean of the auxiliary variable is at 0.

  Returns:
    sampling_op: Tensorflow operation that executes the sampling.
    log_density_ratio: Tensor containing the density ratio of the auxiliary
    variable.
  """
  if aux_variance_ratio > 1. or aux_variance_ratio < 0.:
    raise ValueError(
        "The ratio of the variance of the auxiliary variable must be between 0 "
        "and 1."
    )

  p_a1_loc = tf.zeros_like(prior.loc)
  p_a1_scale = tf.math.sqrt(prior.scale**2 * aux_variance_ratio)
  p_a1 = tfp.distributions.Normal(loc=p_a1_loc, scale=p_a1_scale)
  p_a2_loc = prior.loc
  p_a2_scale = tf.math.sqrt(prior.scale**2 - p_a1_scale**2)
  # q(a1)
  a1_loc = (posterior.loc - prior.loc) * p_a1_scale**2 / prior.scale**2
  a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a1_scale**2 / prior.scale**2 + p_a2_scale**2) *
      p_a1_scale**2 / prior.scale**2)
  q_a1 = tfp.distributions.Normal(loc=a1_loc, scale=a1_scale)
  a1 = q_a1.sample()

  # q(z|a1)
  z_a1_loc = prior.loc + (
      (posterior.loc - prior.loc) * p_a2_scale**2 * prior.scale**2 +
      a1 * posterior.scale**2 * prior.scale**2) / (
          prior.scale**2 * p_a2_scale**2 + posterior.scale**2 * p_a1_scale**2)
  z_a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a2_scale**2 * prior.scale**2) /
      (prior.scale**2 * p_a2_scale**2 + p_a1_scale**2 * posterior.scale**2))

  with tf1.control_dependencies([
      q_a1.loc, q_a1.scale, p_a1.loc, p_a1.scale, a1, p_a2_loc, p_a2_scale,
      z_a1_loc, z_a1_scale
  ]):
    log_density_ratio = q_a1.log_prob(a1) - p_a1.log_prob(a1)
    prior_update = [
        prior.loc.assign(a1 + p_a2_loc),
        prior.untransformed_scale.assign(tfp.math.softplus_inverse(p_a2_scale))
    ]
    posterior_update = [
        posterior.loc.assign(z_a1_loc),
        posterior.untransformed_scale.assign(
            tfp.math.softplus_inverse(z_a1_scale))
    ]
  return [prior_update, posterior_update], tf.reduce_sum(log_density_ratio)
