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

"""Tests for tracers."""

import edward2 as ed
import tensorflow as tf


class TracersTest(tf.test.TestCase):

  def testCondition(self):
    tf.random.set_seed(358758)
    def model():
      x = ed.Normal(loc=-5., scale=1e-8, name="x")
      y = ed.Normal(loc=x, scale=1e-8, name="y")
      return x, y

    with ed.condition(x=5.):
      x, y = model()

    self.assertEqual(x, 5.)
    self.assertAllClose(tf.convert_to_tensor(y), 5., atol=1e-3)

  def testTape(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1., name="y")
      return x + y

    with ed.tape() as model_tape:
      output = model()

    self.assertEqual(list(model_tape.keys()), ["x", "y"])
    expected_value = model_tape["x"] + model_tape["y"]
    actual_value = output
    self.assertEqual(expected_value, actual_value)

  def testTapeNoName(self):
    def model():
      x = ed.Normal(loc=0., scale=1., name="x")
      y = ed.Normal(loc=x, scale=1.)
      return x + y

    with ed.tape() as model_tape:
      _ = model()

    self.assertEqual(list(model_tape.keys()), ["x"])

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

    self.assertEqual(list(model_tape.keys()), ["x", "y"])
    expected_value = 2. * model_tape["x"] + 2. * model_tape["y"]
    actual_value = output
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

    self.assertEqual(list(model_tape.keys()), ["x", "y"])
    expected_value = model_tape["x"] + model_tape["y"]
    actual_value = output
    self.assertEqual(expected_value, actual_value)


if __name__ == "__main__":
  tf.test.main()
