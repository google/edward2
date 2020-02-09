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

"""Tests for utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  def testAddWeightWithTrainableInitializer(self):
    dense_wrapped = ed.layers.utils.add_weight(tf.keras.layers.Dense)
    initializer = ed.initializers.get('trainable_normal')
    layer = dense_wrapped(2, kernel_initializer=initializer)
    inputs = tf.random.normal([1, 3])
    _ = layer(inputs)
    self.assertTrue(initializer.built, True)
    self.assertNotEmpty(initializer.weights)
    for weight in initializer.weights:
      self.assertTrue(np.any([weight is lweight for lweight in layer.weights]))

  def testAddWeightWithTrainableRegularizer(self):
    dense_wrapped = ed.layers.utils.add_weight(tf.keras.layers.Dense)
    regularizer = ed.regularizers.get('trainable_normal_kl_divergence_stddev')
    layer = dense_wrapped(2, kernel_regularizer=regularizer)
    inputs = tf.random.normal([1, 3])
    _ = layer(inputs)
    self.assertTrue(regularizer.built, True)
    self.assertNotEmpty(regularizer.weights)
    for weight in regularizer.weights:
      self.assertTrue(np.any([weight is lweight for lweight in layer.weights]))

  def testOneHotAddExactHard(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    shift = tf.constant([[0., 1., 0.],
                         [1., 0., 0.]])

    outputs = ed.layers.utils.one_hot_add(inputs, shift)
    outputs_val = self.evaluate(outputs)
    self.assertAllClose(outputs_val,
                        np.array([[0., 0., 1.],
                                  [0., 0., 1.]], dtype=np.float32),
                        rtol=1e-4, atol=1e-4)

  def testOneHotMinusExactHard(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    shift = tf.constant([[0., 1., 0.],
                         [1., 0., 0.]])

    outputs = ed.layers.utils.one_hot_minus(inputs, shift)
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val, np.array([[1., 0., 0.],
                                               [0., 0., 1.]], dtype=np.float32))

  def testOneHotMultiplyExactHard(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    scale = tf.constant([[0., 1., 0.],
                         [0., 0., 1.]])

    outputs = ed.layers.utils.one_hot_multiply(inputs, scale)
    outputs_val = self.evaluate(outputs)
    self.assertAllEqual(outputs_val, np.array([[0., 1., 0.],
                                               [0., 1., 0.]], dtype=np.float32))

  def testOneHotAddExactSoft(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    shift = tf.constant([[0.1, 0.6, 0.3],
                         [0.2, 0.4, 0.4]])

    outputs = ed.layers.utils.one_hot_add(inputs, shift)

    shift_zero = inputs
    shift_one = np.array([[0., 0., 1.],
                          [1., 0., 0.]])
    shift_two = np.array([[1., 0., 0.],
                          [0., 1., 0.]])
    expected_outputs = (shift[..., 0][..., tf.newaxis] * shift_zero +
                        shift[..., 1][..., tf.newaxis] * shift_one +
                        shift[..., 2][..., tf.newaxis] * shift_two)

    actual_outputs_val, expected_outputs_val = self.evaluate([
        outputs, expected_outputs])
    self.assertAllClose(actual_outputs_val, expected_outputs_val)

  def testOneHotMinusExactSoft(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    shift = tf.constant([[0.1, 0.6, 0.3],
                         [0.2, 0.4, 0.4]])

    outputs = ed.layers.utils.one_hot_minus(inputs, shift)

    shift_zero = inputs
    shift_one = np.array([[1., 0., 0.],
                          [0., 1., 0.]])
    shift_two = np.array([[0., 0., 1.],
                          [1., 0., 0.]])
    expected_outputs = (shift[..., 0][..., tf.newaxis] * shift_zero +
                        shift[..., 1][..., tf.newaxis] * shift_one +
                        shift[..., 2][..., tf.newaxis] * shift_two)

    actual_outputs_val, expected_outputs_val = self.evaluate([
        outputs, expected_outputs])
    self.assertAllEqual(actual_outputs_val, expected_outputs_val)

  def testOneHotMultiplyExactSoft(self):
    inputs = tf.constant([[0., 1., 0.],
                          [0., 0., 1.]])
    scale = tf.constant([[0.1, 0.6, 0.3],
                         [0.2, 0.4, 0.4]])

    outputs = ed.layers.utils.one_hot_multiply(inputs, scale)

    scale_zero = np.array([[0., 0., 0.],
                           [0., 0., 0.]])
    scale_one = inputs
    scale_two = np.array([[0., 0., 1.],
                          [0., 1., 0.]])
    expected_outputs = (scale[..., 0][..., tf.newaxis] * scale_zero +
                        scale[..., 1][..., tf.newaxis] * scale_one +
                        scale[..., 2][..., tf.newaxis] * scale_two)

    actual_outputs_val, expected_outputs_val = self.evaluate([
        outputs, expected_outputs])
    self.assertAllEqual(actual_outputs_val, expected_outputs_val)

  @parameterized.parameters(
      (ed.layers.utils.one_hot_add,),
      (ed.layers.utils.one_hot_minus,),
  )
  def testOneHotAddShapeHard(self, one_hot_add_fn):
    batch_size = 2
    length = 4
    vocab_size = 5
    inputs = tf.random.uniform(
        [batch_size, length], minval=0, maxval=vocab_size, dtype=tf.int32)
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    shift = tf.random.uniform(
        [batch_size, length], minval=0, maxval=vocab_size, dtype=tf.int32)
    shift = tf.one_hot(shift, depth=vocab_size)

    outputs = one_hot_add_fn(inputs, shift)
    outputs_val = self.evaluate(outputs)
    self.assertEqual(outputs_val.shape, (batch_size, length, vocab_size))

  @parameterized.parameters(
      (ed.layers.utils.one_hot_add,),
      (ed.layers.utils.one_hot_minus,),
  )
  def testOneHotAddShapeSoft(self, one_hot_add_fn):
    batch_size = 2
    length = 4
    vocab_size = 5
    inputs = tf.random.uniform([batch_size, length, vocab_size])
    shift = tf.random.uniform([batch_size, length, vocab_size])

    outputs = one_hot_add_fn(inputs, shift)
    outputs_val = self.evaluate(outputs)
    self.assertEqual(outputs_val.shape, (batch_size, length, vocab_size))

  def testMultiplicativeInverse(self):
    batch_size = 3
    vocab_size = 79
    length = 5
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    one_hot_inputs = tf.one_hot(inputs, depth=vocab_size)

    one_hot_inv = ed.layers.utils.multiplicative_inverse(one_hot_inputs,
                                                         vocab_size)
    inv_inputs = tf.argmax(one_hot_inv, axis=-1)
    inputs_inv_inputs = tf.math.floormod(inputs * inv_inputs, vocab_size)
    inputs_inv_inputs_val = self.evaluate(inputs_inv_inputs)
    self.assertAllEqual(inputs_inv_inputs_val, np.ones((batch_size, length)))

  def testApproximatelyStochastic(self):
    rng = np.random.RandomState(0)
    tf.random.set_seed(1)
    for dims in [2, 5, 10]:
      for batch_size in [1, 2, 10]:
        log_alpha = rng.randn(batch_size, dims, dims)
        result = ed.layers.utils.sinkhorn(log_alpha)
        result_val = self.evaluate(result)
        self.assertAllClose(np.sum(result_val, 1),
                            np.tile([1.0], (batch_size, dims)),
                            atol=1e-3)
        self.assertAllClose(np.sum(result_val, 2),
                            np.tile([1.0], (batch_size, dims)),
                            atol=1e-3)

  def testSoftToHardPermutation(self):
    """The solution of the matching for the identity matrix is range(N)."""
    dims = 10
    identity = tf.eye(dims)
    result_matching = ed.layers.utils.soft_to_hard_permutation(identity)
    result_matching_val = self.evaluate(result_matching)
    self.assertAllEqual(result_matching_val[0], np.eye(dims))


if __name__ == '__main__':
  tf.test.main()
