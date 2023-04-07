# coding=utf-8
# Copyright 2023 The Edward2 Authors.
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

"""Tests for generated random variables."""

import inspect
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class GeneratedRandomVariablesTest(parameterized.TestCase, tf.test.TestCase):

  def testBernoulliDoc(self):
    self.assertGreater(len(ed.Bernoulli.__doc__), 0)
    self.assertIn(
        inspect.cleandoc(tfp.distributions.Bernoulli.__init__.__doc__),
        ed.Bernoulli.__doc__)
    self.assertEqual(ed.Bernoulli.__name__, "Bernoulli")

  @parameterized.named_parameters(
      {"testcase_name": "1d_rv_1d_event", "logits": np.zeros(1), "n": [1]},
      {"testcase_name": "1d_rv_5d_event", "logits": np.zeros(1), "n": [5]},
      {"testcase_name": "5d_rv_1d_event", "logits": np.zeros(5), "n": [1]},
      {"testcase_name": "5d_rv_5d_event", "logits": np.zeros(5), "n": [5]},
  )
  def testBernoulliLogProb(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfp.distributions.Bernoulli(logits)
    x = rv.distribution.sample(n)
    self.assertAllEqual(rv.distribution.log_prob(x), dist.log_prob(x))

  @parameterized.named_parameters(
      {"testcase_name": "0d_rv_0d_sample",
       "logits": 0.,
       "n": 1},
      {"testcase_name": "0d_rv_1d_sample",
       "logits": 0.,
       "n": [1]},
      {"testcase_name": "1d_rv_1d_sample",
       "logits": np.array([0.]),
       "n": [1]},
      {"testcase_name": "1d_rv_5d_sample",
       "logits": np.array([0.]),
       "n": [5]},
      {"testcase_name": "2d_rv_1d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [1]},
      {"testcase_name": "2d_rv_5d_sample",
       "logits": np.array([-0.2, 0.8]),
       "n": [5]},
  )
  def testBernoulliSample(self, logits, n):
    rv = ed.Bernoulli(logits)
    dist = tfp.distributions.Bernoulli(logits)
    self.assertEqual(rv.distribution.sample(n).shape, dist.sample(n).shape)

  @parameterized.named_parameters(
      {"testcase_name": "0d_bernoulli",
       "rv_cls": ed.Bernoulli,
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": [],
       "probs": 0.5},
      {"testcase_name": "2d_bernoulli",
       "rv_cls": ed.Bernoulli,
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": [],
       "logits": np.zeros([2, 3])},
      {"testcase_name": "2x0d_bernoulli",
       "rv_cls": ed.Bernoulli,
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": [],
       "probs": 0.5},
      {"testcase_name": "2x1d_bernoulli",
       "rv_cls": ed.Bernoulli,
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": [],
       "probs": 0.5},
      {"testcase_name": "3d_dirichlet",
       "rv_cls": ed.Dirichlet,
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": [3],
       "concentration": np.ones(3)},
      {"testcase_name": "2x3d_dirichlet",
       "rv_cls": ed.Dirichlet,
       "sample_shape": [],
       "batch_shape": [2],
       "event_shape": [3],
       "concentration": np.ones([2, 3])},
      {"testcase_name": "1x3d_dirichlet",
       "rv_cls": ed.Dirichlet,
       "sample_shape": [1],
       "batch_shape": [],
       "event_shape": [3],
       "concentration": np.ones(3)},
      {"testcase_name": "2x1x3d_dirichlet",
       "rv_cls": ed.Dirichlet,
       "sample_shape": [2, 1],
       "batch_shape": [],
       "event_shape": [3],
       "concentration": np.ones(3)},
  )
  def testShape(self, rv_cls, sample_shape, batch_shape, event_shape, **kwargs):
    rv = rv_cls(sample_shape=sample_shape, **kwargs)
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2, "loc": 0.5, "scale": 1.0},
      {"cls": ed.Normal, "value": [2], "loc": [0.5], "scale": [1.0]},
      {"cls": ed.Poisson, "value": 2, "rate": 0.5},
  )
  def testValueShapeAndDtype(self, cls, value, **kwargs):
    rv = cls(value=value, **kwargs)
    value_shape = rv.value.shape
    expected_shape = rv.sample_shape.concatenate(
        rv.distribution.batch_shape).concatenate(rv.distribution.event_shape)
    self.assertEqual(value_shape, expected_shape)
    self.assertEqual(rv.distribution.dtype, rv.value.dtype)

  @parameterized.parameters(
      {"cls": ed.Normal, "value": 2, "loc": [0.5, 0.5], "scale": 1.0},
      {"cls": ed.Normal, "value": 2, "loc": [0.5], "scale": [1.0]},
      {"cls": ed.Normal, "value": np.zeros([10, 3]), "loc": [0.5, 0.5],
       "scale": [1.0, 1.0]},
  )
  def testValueMismatchRaises(self, cls, value, **kwargs):
    with self.assertRaises(ValueError):
      cls(value=tf.convert_to_tensor(value), **kwargs)

  def testMakeRandomVariable(self):
    """Tests that manual wrapping is the same as the built-in solution."""
    custom_normal = ed.make_random_variable(tfp.distributions.Normal)

    def model_builtin():
      return ed.Normal(1., 0.1, name="x")

    def model_wrapped():
      return custom_normal(1., 0.1, name="x")

    log_joint_builtin = ed.make_log_joint_fn(model_builtin)
    log_joint_wrapped = ed.make_log_joint_fn(model_wrapped)
    self.assertEqual(log_joint_builtin(x=7.), log_joint_wrapped(x=7.))

if __name__ == "__main__":
  tf.test.main()
