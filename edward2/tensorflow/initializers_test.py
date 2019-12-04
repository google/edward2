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

"""Tests for Keras-style initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class InitializersTest(tf.test.TestCase):

  def testTrainableDeterministic(self):
    tf.random.set_seed(345689)
    shape = (100,)
    initializer = ed.initializers.get('trainable_deterministic')
    rv = initializer(shape)
    self.evaluate(tf1.global_variables_initializer())
    # Get distribution of rv -> get distribution of Independent.
    loc_value = self.evaluate(rv.distribution.distribution.loc)
    self.assertNotAllClose(loc_value, np.zeros(shape))

    rv_value = self.evaluate(rv)
    self.assertEqual(rv_value.shape, shape)

  def testTrainableHalfCauchy(self):
    tf.random.set_seed(2832)
    shape = (3,)
    initializer = ed.initializers.get('trainable_half_cauchy')
    half_cauchy = initializer(shape)
    self.evaluate(tf1.global_variables_initializer())
    loc_value, scale_value = self.evaluate([
        # Get distribution of rv -> get distribution of Independent.
        half_cauchy.distribution.distribution.loc,
        half_cauchy.distribution.distribution.scale])
    self.assertAllClose(loc_value, np.zeros(shape), atol=1e-4)
    target_scale = np.log(1. + np.exp(-3.))
    self.assertAllClose(scale_value, target_scale * np.ones(shape), atol=5e-2)

    half_cauchy_value = self.evaluate(half_cauchy)
    self.assertAllEqual(half_cauchy_value.shape, shape)
    self.assertAllGreaterEqual(half_cauchy_value, 0.)

  def testTrainableNormal(self):
    tf.random.set_seed(345689)
    shape = (100,)
    initializer = ed.initializers.get('trainable_normal')
    normal = initializer(shape)
    self.evaluate(tf1.global_variables_initializer())
    loc_value, scale_value = self.evaluate([
        # Get distribution of rv -> get distribution of Independent.
        normal.distribution.distribution.loc,
        normal.distribution.distribution.scale])
    self.assertAllClose(loc_value, np.zeros(shape), atol=1e-4)
    target_scale = np.log(1. + np.exp(-3.))
    self.assertAllClose(scale_value, target_scale * np.ones(shape), atol=5e-2)

    normal_value = self.evaluate(normal)
    self.assertEqual(normal_value.shape, shape)

  def testTrainableMixtureOfDeltas(self):
    tf.random.set_seed(345689)
    shape = (100,)
    num_components = 5
    initializer = ed.initializers.TrainableMixtureOfDeltas(num_components)
    mixture_shape = list(shape) + [num_components]
    rv = initializer(shape)
    self.evaluate(tf1.global_variables_initializer())
    probs_value, loc_value = self.evaluate([
        # Get distribution of rv -> get distribution of Independent.
        rv.distribution.distribution.mixture_distribution.probs,
        rv.distribution.distribution.components_distribution.loc,
    ])
    self.assertAllClose(
        probs_value,
        tf.broadcast_to([[1/num_components]*num_components], mixture_shape),
        atol=1e-4)
    self.assertAllClose(loc_value, np.zeros(mixture_shape), atol=1.)

    value = self.evaluate(rv)
    self.assertEqual(value.shape, shape)

  def testInitializersGet(self):
    self.assertIsInstance(ed.initializers.get('trainable_normal'),
                          ed.initializers.TrainableNormal)
    # This is working correctly, but the test won't pass currently because TF
    # isn't consistent (yet).  Specifically, tf.keras.initializers.get('zeros')
    # returns a certain class while tf.keras.initializers.zeros (or Zeros)
    # currently returns v2 of that class.
    # self.assertIsInstance(ed.initializers.get('zeros'),
    #                       tf.keras.initializers.Zeros().__class__)


if __name__ == '__main__':
  tf.test.main()
