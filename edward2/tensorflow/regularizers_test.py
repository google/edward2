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

"""Tests for Keras-style regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class RegularizersTest(parameterized.TestCase, tf.test.TestCase):

  def testCauchyKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('cauchy_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    kl_value = self.evaluate(kl)
    # KL uses a single-sample estimate, which is not necessarily >0. We only
    # check shape.
    self.assertEqual(kl_value.shape, ())

  def testHalfCauchyKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('half_cauchy_kl_divergence')
    variational_posterior = ed.Independent(
        ed.LogNormal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    kl_value = self.evaluate(kl)
    # KL uses a single-sample estimate, which is not necessarily >0. We only
    # check shape.
    self.assertEqual(kl_value.shape, ())

  def testNormalKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('normal_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    kl_value = self.evaluate(kl)
    self.assertGreaterEqual(kl_value, 0.)

    dataset_size = 100
    scale_factor = 1. / dataset_size
    regularizer = ed.regularizers.NormalKLDivergence(scale_factor=scale_factor)
    kl = regularizer(variational_posterior)
    scaled_kl_value = self.evaluate(kl)
    self.assertEqual(scale_factor * kl_value, scaled_kl_value)

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
    kl_value, eb_kl_value = self.evaluate([kl, eb_kl])
    self.assertGreaterEqual(kl_value, eb_kl_value)
    self.assertAlmostEqual(kl_value, eb_kl_value, delta=0.05,
                           msg='Parameters score KL=%.6f on generating '
                           'Normal-IG KL and KL=%.6f on EB-fitted KL, '
                           'too much difference.' % (kl_value, eb_kl_value))

  def testNormalEmpiricalBayesKLDivergenceTFFunction(self):
    """Checks that KL evaluates properly multiple times when compiled."""
    shape = (3,)
    regularizer = ed.regularizers.get('normal_empirical_bayes_kl_divergence')
    regularizer_compiled = tf.function(regularizer)
    weights_one = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=len(shape))
    kl_one = regularizer(weights_one)
    kl_one_c = regularizer_compiled(weights_one)

    weights_two = ed.Independent(
        ed.Normal(loc=5. + tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=len(shape))
    kl_two = regularizer(weights_two)
    kl_two_c = regularizer_compiled(weights_two)

    kl_one_value, kl_one_c_value, kl_two_value, kl_two_c_value = self.evaluate(
        [kl_one, kl_one_c, kl_two, kl_two_c])
    self.assertAllClose(kl_one_value, kl_one_c_value)
    self.assertAllClose(kl_two_value, kl_two_c_value)
    self.assertNotAlmostEqual(kl_one_c_value, kl_two_c_value)

  def testTrainableNormalKLDivergenceStddev(self):
    tf.random.set_seed(83271)
    shape = (3,)
    regularizer = ed.regularizers.get('trainable_normal_kl_divergence_stddev')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    self.evaluate(tf1.global_variables_initializer())
    kl_value = self.evaluate(kl)
    self.assertGreaterEqual(kl_value, 0.)

    prior_stddev = self.evaluate(
        regularizer.stddev_constraint(regularizer.stddev))
    self.assertAllClose(prior_stddev, np.ones(prior_stddev.shape),
                        atol=0.1)

  def testUniformKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('uniform_kl_divergence')
    variational_posterior = ed.Independent(
        ed.Normal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    kl_value = self.evaluate(kl)
    self.assertNotEqual(kl_value, 0.)

    dataset_size = 100
    scale_factor = 1. / dataset_size
    regularizer = ed.regularizers.UniformKLDivergence(scale_factor=scale_factor)
    kl = regularizer(variational_posterior)
    scaled_kl_value = self.evaluate(kl)
    self.assertAlmostEqual(scale_factor * kl_value, scaled_kl_value)

  def testRegularizersGet(self):
    self.assertIsInstance(ed.regularizers.get('normal_kl_divergence'),
                          ed.regularizers.NormalKLDivergence)
    self.assertIsInstance(ed.regularizers.get('l2'), tf.keras.regularizers.L1L2)

if __name__ == '__main__':
  tf.test.main()
