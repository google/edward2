# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Tests for random variable."""

import re
from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class FakeDistribution(tfp.distributions.Distribution):
  """Fake distribution class for testing."""

  def __init__(self):
    super(FakeDistribution, self).__init__(
        dtype=tf.float32,
        reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True)

  def _sample_n(self, *args, **kwargs):
    return tf.ones(shape=(4, 4))

  def sample(self, *args, **kwargs):
    return tf.ones(shape=(4, 4))


class FakeDistributionNoSample(tfp.distributions.Distribution):
  """Fake distribution class for testing."""

  def __init__(self):
    super(FakeDistributionNoSample, self).__init__(
        dtype=None,
        reparameterization_type=tfp.distributions.FULLY_REPARAMETERIZED,
        validate_args=False,
        allow_nan_stats=True)


class RandomVariableTest(parameterized.TestCase, tf.test.TestCase):

  def testConstructor(self):
    x = ed.RandomVariable(tfp.distributions.Poisson(rate=np.ones([2, 5])),
                          value=np.ones([2, 5]))
    self.assertAllEqual(tf.convert_to_tensor(x), x.value)
    with self.assertRaises(ValueError):
      _ = ed.RandomVariable(tfp.distributions.Bernoulli(probs=0.5),
                            value=tf.zeros([2, 5], dtype=tf.int32))
    x = ed.RandomVariable(FakeDistribution())
    with self.assertRaises(NotImplementedError):
      _ = ed.RandomVariable(FakeDistributionNoSample())

  def testTraceType(self):
    x_1 = ed.RandomVariable(tfp.distributions.Normal(0., 1.))
    x_2 = ed.RandomVariable(tfp.distributions.Normal(0., 1.))
    trace_type_1 = x_1.__tf_tracing_type__(None)
    trace_type_2 = x_2.__tf_tracing_type__(None)

    self.assertEqual(trace_type_1, trace_type_1)
    self.assertTrue(trace_type_1.is_subtype_of(trace_type_1))

    self.assertNotEqual(trace_type_1, trace_type_2)
    self.assertFalse(trace_type_1.is_subtype_of(trace_type_2))

  def testGradientsFirstOrder(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0., 1.))
    def f(x):
      return 2. * x
    with tf.GradientTape() as tape:
      tape.watch(x.value)
      y = f(x)
    z = tape.gradient(y, [x.value])[0]
    self.assertEqual(z, 2.)

  def testGradientsSecondOrder(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    def f(x):
      return 2 * (x ** 2)
    with tf.GradientTape() as tape2:
      tape2.watch(x.value)
      with tf.GradientTape() as tape:
        tape.watch(x.value)
        y = f(x)
      z = tape.gradient(y, [x.value])[0]
    z = tape2.gradient(z, [x.value])[0]
    self.assertEqual(z, 4.0)

  def testStr(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0), value=1.234)
    pattern = "RandomVariable(\"1.234\", shape=(), dtype=float32"
    regexp = re.escape(pattern)
    self.assertRegexpMatches(str(x), regexp)

  def testRepr(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0), value=1.234)
    string = (
        "<ed.RandomVariable '{name}' shape=() "
        "dtype=float32 numpy=np.float32(1.234)>".format(
            name=x.distribution.name
        )
    )
    self.assertEqual(repr(x), string)

  def testNumpy(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0), value=1.23)
    self.assertEqual(x, tf.constant(1.23))

  def testOperatorsAdd(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x + y
    z_value = x.value + y
    self.assertAllEqual(z, z_value)

  def testOperatorsRadd(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y + x
    z_value = y + x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsSub(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x - y
    z_value = x.value - y
    self.assertAllEqual(z, z_value)

  def testOperatorsRsub(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y - x
    z_value = y - x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsMul(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x * y
    z_value = x.value * y
    self.assertAllEqual(z, z_value)

  def testOperatorsRmul(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y * x
    z_value = y * x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsDiv(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x / y
    z_value = x.value / y
    self.assertAllEqual(z, z_value)

  def testOperatorsRdiv(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y / x
    z_value = y / x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsFloordiv(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x // y
    z_value = x.value // y
    self.assertAllEqual(z, z_value)

  def testOperatorsRfloordiv(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y // x
    z_value = y // x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsMod(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x % y
    z_value = x.value % y
    self.assertAllEqual(z, z_value)

  def testOperatorsRmod(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y % x
    z_value = y % x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsLt(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x < y
    z_value = x.value < y
    self.assertAllEqual(z, z_value)

  def testOperatorsLe(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x <= y
    z_value = x.value <= y
    self.assertAllEqual(z, z_value)

  def testOperatorsGt(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x > y
    z_value = x.value > y
    self.assertAllEqual(z, z_value)

  def testOperatorsGe(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x >= y
    z_value = x.value >= y
    self.assertAllEqual(z, z_value)

  def testOperatorsGetitem(self):
    x = ed.RandomVariable(tfp.distributions.Normal(tf.random.normal([3, 4]),
                                                   1.))
    z = x[0:2, 2:3]
    z_value = x.value[0:2, 2:3]
    self.assertIsInstance(z, ed.RandomVariable)
    self.assertAllEqual(z.distribution.mean(), x.distribution.mean()[0:2, 2:3])
    self.assertAllEqual(tf.convert_to_tensor(z), z_value)

  def testOperatorsPow(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = x ** y
    z_value = x.value ** y
    self.assertAllEqual(z, z_value)

  def testOperatorsRpow(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    z = y ** x
    z_value = y ** x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsNeg(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    z = -x
    z_value = -x.value
    self.assertAllEqual(z, z_value)

  def testOperatorsAbs(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    z = abs(x)
    z_value = abs(x.value)
    self.assertAllEqual(z, z_value)

  def testOperatorsHash(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(hash(x), hash(y))
    self.assertEqual(hash(x), id(x))

  # TODO(trandustin): Re-enable test.
  # def testOperatorsEq(self):
  #   x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
  #   self.assertEqual(x, x)

  def testOperatorsNe(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = 5.0
    self.assertNotEqual(x, y)

  def testOperatorsBoolNonzero(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    with self.assertRaises(TypeError):
      _ = not x

  def testArrayPriority(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 1.0))
    y = np.array(5.0, dtype=np.float32)
    z = y / x
    z_value = y / x.value
    self.assertAllEqual(z, z_value)

  def testConvertToTensor(self):
    x = ed.RandomVariable(tfp.distributions.Normal(0.0, 0.1))
    with self.assertRaises(ValueError):
      _ = tf.convert_to_tensor(x, dtype=tf.int32)

  @parameterized.parameters(
      {"probs": 0.5,
       "sample_shape": [],
       "batch_shape": [],
       "event_shape": []},
      {"probs": np.zeros([2, 3]) + 0.5,
       "sample_shape": [],
       "batch_shape": [2, 3],
       "event_shape": []},
      {"probs": 0.5,
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"probs": 0.5,
       "sample_shape": [2],
       "batch_shape": [],
       "event_shape": []},
      {"probs": 0.5,
       "sample_shape": [2, 4],
       "batch_shape": [],
       "event_shape": []},
  )
  def testShape(self, probs, sample_shape, batch_shape, event_shape):
    rv = ed.RandomVariable(tfp.distributions.Bernoulli(probs=probs),
                           sample_shape=sample_shape)
    self.assertEqual(rv.shape, sample_shape + batch_shape + event_shape)
    self.assertEqual(rv.shape, rv.shape)
    self.assertEqual(rv.sample_shape, sample_shape)
    self.assertEqual(rv.distribution.batch_shape, batch_shape)
    self.assertEqual(rv.distribution.event_shape, event_shape)

  def testRandomTensorSample(self):
    num_samples = tf.cast(tfp.distributions.Poisson(rate=5.).sample(), tf.int32)
    _ = ed.RandomVariable(tfp.distributions.Normal(loc=0.0, scale=1.0),
                          sample_shape=num_samples)


if __name__ == "__main__":
  tf.test.main()
