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

"""Tests for Keras-style regularizers."""

import itertools
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class RegularizersTest(parameterized.TestCase, tf.test.TestCase):

  def testCauchyKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('cauchy_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    # KL uses a single-sample estimate, which is not necessarily >0. We only
    # check shape.
    self.assertEqual(kl.shape, ())

  def testHalfCauchyKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('half_cauchy_kl_divergence')
    variational_posterior = ed.Independent(
        ed.LogNormal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    # KL uses a single-sample estimate, which is not necessarily >0. We only
    # check shape.
    self.assertEqual(kl.shape, ())

  def testLogNormalKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('log_normal_kl_divergence')
    variational_posterior = ed.Independent(
        ed.LogNormal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    self.assertGreaterEqual(kl, 0.)

    dataset_size = 100
    scale_factor = 1. / dataset_size
    regularizer = ed.regularizers.LogNormalKLDivergence(
        scale_factor=scale_factor)
    scaled_kl = regularizer(variational_posterior)
    self.assertEqual(scale_factor * kl, scaled_kl)

  def testNormalKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('normal_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    self.assertGreaterEqual(kl, 0.)

    dataset_size = 100
    scale_factor = 1. / dataset_size
    regularizer = ed.regularizers.NormalKLDivergence(scale_factor=scale_factor)
    scaled_kl = regularizer(variational_posterior)
    self.assertEqual(scale_factor * kl, scaled_kl)

  @parameterized.parameters(
      itertools.product(
          [0.1, 1.0, 2.0],
          [0.1, 1.0, 2.0]))
  def testNormalEmpiricalBayesKLDivergence(self, gen_stddev, eb_prior_stddev):
    """Tests ed.regularizers.NormalEmpiricalBayesKLDivergence.

    Check that EB KL estimate should always be smaller but close to the true
    generating Normal-InverseGamma KL due to it being explicitly optimized.

    Args:
      gen_stddev: Standard deviation of the generating normal distribution.
      eb_prior_stddev: Standard deviation of the EB hyperprior.
    """
    tf.random.set_seed(89323)
    shape = (99, 101)
    gen_mean = 0.
    eb_prior_mean = eb_prior_stddev**2
    cvar = (eb_prior_mean / eb_prior_stddev)**2
    variance_concentration = cvar + 2.
    variance_scale = eb_prior_mean*(cvar + 1.)
    weight = ed.Independent(
        ed.Normal(gen_mean + tf.zeros(shape), gen_stddev).distribution,
        reinterpreted_batch_ndims=len(shape))

    # Compute KL(q(w)|| N(w|gen_stddev)) - log IG(gen_stddev**2) under a fixed
    # setting of the prior stddev.
    normal_regularizer = ed.regularizers.NormalKLDivergence(mean=gen_mean,
                                                            stddev=gen_stddev)
    kl = normal_regularizer(weight)
    kl -= tf.reduce_sum(
        ed.InverseGamma(variance_concentration,
                        variance_scale).distribution.log_prob(gen_stddev**2))

    eb_regularizer = ed.regularizers.NormalEmpiricalBayesKLDivergence(
        mean=gen_mean,
        variance_concentration=variance_concentration,
        variance_scale=variance_scale)
    eb_kl = eb_regularizer(weight)
    # Normalize comparison by total number of weights. (Note this also scales
    # the IG log prob.)
    kl /= float(np.prod(shape))
    eb_kl /= float(np.prod(shape))
    self.assertGreaterEqual(kl, eb_kl)
    self.assertAlmostEqual(kl.numpy(), eb_kl.numpy(), delta=0.05,
                           msg='Parameters score KL=%.6f on generating '
                           'Normal-IG KL and KL=%.6f on EB-fitted KL, '
                           'too much difference.' % (kl, eb_kl))

  def testNormalEmpiricalBayesKLDivergenceTFFunction(self):
    """Checks that KL evaluates properly multiple times when compiled."""
    shape = (3,)
    regularizer = ed.regularizers.get('normal_empirical_bayes_kl_divergence')
    regularizer_compiled = tf.function(regularizer)
    weights_one = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=len(shape))
    kl_one = regularizer(weights_one).numpy()
    kl_one_c = regularizer_compiled(weights_one).numpy()

    weights_two = ed.Independent(
        ed.Normal(loc=5. + tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=len(shape))
    kl_two = regularizer(weights_two).numpy()
    kl_two_c = regularizer_compiled(weights_two).numpy()

    self.assertAllClose(kl_one, kl_one_c)
    self.assertAllClose(kl_two, kl_two_c)
    self.assertNotAlmostEqual(kl_one_c, kl_two_c)

  def testTrainableNormalKLDivergenceStddev(self):
    tf.random.set_seed(83271)
    shape = (3,)
    regularizer = ed.regularizers.get('trainable_normal_kl_divergence_stddev')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    self.assertGreaterEqual(kl, 0.)

    prior_stddev = regularizer.stddev_constraint(regularizer.stddev)
    self.assertAllClose(prior_stddev, np.ones(prior_stddev.shape),
                        atol=0.1)

  def testUniformKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('uniform_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    self.assertNotEqual(kl, 0.)

    dataset_size = 100
    scale_factor = 1. / dataset_size
    regularizer = ed.regularizers.UniformKLDivergence(scale_factor=scale_factor)
    scaled_kl = regularizer(variational_posterior)
    self.assertAlmostEqual(scale_factor * kl, scaled_kl)

  def testRegularizersGet(self):
    self.assertIsInstance(ed.regularizers.get('normal_kl_divergence'),
                          ed.regularizers.NormalKLDivergence)
    self.assertIsInstance(ed.regularizers.get('l2'), tf.python.keras.regularizers.L2)
    self.assertIsNone(ed.regularizers.get(''))

if __name__ == '__main__':
  tf.test.main()
