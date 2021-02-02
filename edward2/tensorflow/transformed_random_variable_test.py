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

"""Tests for transformed random variables."""

import edward2 as ed
import numpy as np
import tensorflow as tf


class TransformedRandomVariableTest(tf.test.TestCase):

  def testTransformedRandomVariable(self):
    class Exp(tf.python.keras.layers.Layer):
      """Exponential activation function for reversible networks."""

      def __call__(self, inputs, *args, **kwargs):
        if not isinstance(inputs, ed.RandomVariable):
          return super(Exp, self).__call__(inputs, *args, **kwargs)
        return ed.TransformedRandomVariable(inputs, self)

      def call(self, inputs):
        return tf.exp(inputs)

      def reverse(self, inputs):
        return tf.math.log(inputs)

      def log_det_jacobian(self, inputs):
        return -tf.math.log(inputs)

    x = ed.Normal(0., 1.)
    y = Exp()(x)
    y_sample = y.distribution.sample()
    y_log_prob = y.distribution.log_prob(y_sample)
    self.assertGreater(y_sample, 0.)
    self.assertTrue(np.isfinite(y_log_prob))


if __name__ == '__main__':
  tf.test.main()
