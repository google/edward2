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

"""Tests for tracers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import six
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class TracersTest(tf.test.TestCase):

  def testCondition(self):
    tf.random.set_seed(358758)
    def model():
      x = ed.Normal(loc=-5., scale=1e-8, name="x")
      y = ed.Normal(loc=x, scale=1e-8, name="y")
      return x, y

    with ed.condition(x=5.):
      x, y = model()

    x_value, y_value = self.evaluate([x, y])
    self.assertEqual(x_value, 5.)
    self.assertAllClose(y_value, 5., atol=1e-3)

  def testTape(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.tape() as model_tape:
      output = model()

    expected_value, actual_value = self.evaluate([
        model_tape["x"] + model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)

  def testTapeNoName(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1.)
      return x + y

    with ed.tape() as model_tape:
      _ = model()

    self.assertEqual(list(six.iterkeys(model_tape)), ["x"])

  def testTapeOuterForwarding(self):
    def double(f, *args, **kwargs):
      return 2. * ed.traceable(f)(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.tape() as model_tape:
      with ed.trace(double):
        output = model()

    expected_value, actual_value = self.evaluate([
        2. * model_tape["x"] + 2. * model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)

  def testTapeInnerForwarding(self):
    def double(f, *args, **kwargs):
      return 2. * ed.traceable(f)(*args, **kwargs)

    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.trace(double):
      with ed.tape() as model_tape:
        output = model()

    expected_value, actual_value = self.evaluate([
        model_tape["x"] + model_tape["y"], output])
    self.assertEqual(list(six.iterkeys(model_tape)), ["x", "y"])
    self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
  tf.test.main()
