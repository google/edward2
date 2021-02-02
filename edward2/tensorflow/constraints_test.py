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

"""Tests for Keras-style constraints."""

from absl.testing import parameterized
import edward2 as ed
import tensorflow as tf


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
    self.assertAllGreater(constrained_weight, 0.)

  def testConstraintsGet(self):
    self.assertIsInstance(ed.constraints.get('positive'),
                          ed.constraints.Positive)
    self.assertIsInstance(ed.constraints.get('non_neg'),
                          tf.python.keras.constraints.NonNeg)


if __name__ == '__main__':
  tf.test.main()
