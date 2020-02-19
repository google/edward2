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

"""Tests for sampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.experimental.auxiliary_sampling import sampling
import numpy as np
import tensorflow.compat.v1 as tf  # tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class SamplingTest(tf.test.TestCase):

  def _softplus_inverse_np(self, x):
    return np.log(np.exp(x) - 1.)

  def test_mean_field_fn(self):
    p_fn, q_fn = sampling.mean_field_fn()
    layer = tfp.layers.DenseLocalReparameterization(
        100,
        kernel_prior_fn=p_fn,
        kernel_posterior_fn=q_fn,
        bias_prior_fn=p_fn,
        bias_posterior_fn=q_fn)
    self.assertIsInstance(layer, tfp.layers.DenseLocalReparameterization)

  def test_sample_auxiliary_op(self):
    p_fn, q_fn = sampling.mean_field_fn()
    p = p_fn(tf.float32, (), 'test_prior', True, tf.get_variable).distribution
    q = q_fn(tf.float32, (), 'test_posterior', True,
             tf.get_variable).distribution

    # Test benign auxiliary variable
    sample_op, _ = sampling.sample_auxiliary_op(p, q, 1e-10)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    p.loc.load(1., session=sess)
    p.untransformed_scale.load(self._softplus_inverse_np(1.), session=sess)
    q.loc.load(1.1, session=sess)
    q.untransformed_scale.load(self._softplus_inverse_np(0.5), session=sess)
    print(sess.run(q.scale))

    sess.run(sample_op)

    tolerance = 0.0001
    self.assertLess(np.abs(sess.run(p.scale) - 1.), tolerance)
    self.assertLess(np.abs(sess.run(p.loc) - 1.), tolerance)
    self.assertLess(np.abs(sess.run(q.scale) - 0.5), tolerance)
    self.assertLess(np.abs(sess.run(q.loc) - 1.1), tolerance)

    # Test fully determining auxiliary variable
    sample_op, _ = sampling.sample_auxiliary_op(p, q, 1. - 1e-10)
    sess.run(tf.initialize_all_variables())
    p.loc.load(1., session=sess)
    p.untransformed_scale.load(self._softplus_inverse_np(1.), session=sess)
    q.loc.load(1.1, session=sess)
    q.untransformed_scale.load(self._softplus_inverse_np(.5), session=sess)

    sess.run(sample_op)

    self.assertLess(np.abs(sess.run(q.loc) - sess.run(p.loc)), tolerance)
    self.assertLess(sess.run(p.scale), tolerance)
    self.assertLess(sess.run(q.scale), tolerance)

    # Test delta posterior
    sample_op, _ = sampling.sample_auxiliary_op(p, q, 0.5)
    sess.run(tf.initialize_all_variables())
    p.loc.load(1., session=sess)
    p.untransformed_scale.load(self._softplus_inverse_np(1.), session=sess)
    q.loc.load(1.1, session=sess)
    q.untransformed_scale.load(self._softplus_inverse_np(1e-10), session=sess)

    sess.run(sample_op)

    self.assertLess(np.abs(sess.run(q.loc) - 1.1), tolerance)
    self.assertLess(sess.run(q.scale), tolerance)

    # Test prior is posterior
    sample_op, _ = sampling.sample_auxiliary_op(p, q, 0.5)
    sess.run(tf.initialize_all_variables())
    p.loc.load(1., session=sess)
    p.untransformed_scale.load(self._softplus_inverse_np(1.), session=sess)
    q.loc.load(1., session=sess)
    q.untransformed_scale.load(self._softplus_inverse_np(1.), session=sess)

    sess.run(sample_op)

    self.assertLess(np.abs(sess.run(q.loc - p.loc)), tolerance)
    self.assertLess(np.abs(sess.run(q.scale - p.scale)), tolerance)


def softplus_inverse_test(self):
  sess = tf.Session()
  self.assertEqual(sess.run(sampling.softplus_inverse(tf.nn.softplus(5.))), 5.)


if __name__ == '__main__':
  tf.test.main()
