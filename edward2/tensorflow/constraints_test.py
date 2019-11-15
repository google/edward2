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

"""Tests for Keras-style constraints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class ConstraintsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'name': 'exp'},
      {'name': 'positive'},
      {'name': 'softplus'},
  )
  def testPositiveConstraint(self, name):
    weight = tf.random.normal((3,))
    constraint = ed.constraints.get(name)
    constrained_weight = constraint(weight)
    constrained_weight_value = self.evaluate(constrained_weight)
    self.assertAllGreater(constrained_weight_value, 0.)


if __name__ == '__main__':
  tf.test.main()
