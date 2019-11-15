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


if __name__ == '__main__':
  tf.test.main()
