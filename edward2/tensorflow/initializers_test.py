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

"""Tests for Keras-style initializers."""

import edward2 as ed
import numpy as np
import tensorflow as tf


class InitializersTest(tf.test.TestCase):

  def testTrainableDeterministic(self):
    tf.random.set_seed(345689)
    shape = (100,)
    initializer = ed.initializers.get('trainable_deterministic')
    rv = initializer(shape)
    # Get distribution of rv -> get distribution of Independent.
    loc = rv.distribution.distribution.loc
    atol = np.sqrt(6/sum(shape)) + 1e-8
    self.assertAllClose(tf.convert_to_tensor(loc),
                        np.zeros(shape),
                        atol=atol)

    self.assertEqual(rv.shape, shape)

  def testTrainableHalfCauchy(self):
    tf.random.set_seed(2832)
    shape = (3,)
    initializer = ed.initializers.get('trainable_half_cauchy')
    half_cauchy = initializer(shape)
    # Get distribution of rv -> get distribution of Independent.
    loc = half_cauchy.distribution.distribution.loc
    scale = half_cauchy.distribution.distribution.scale
    self.assertAllClose(tf.convert_to_tensor(loc), np.zeros(shape), atol=1e-4)
    target_scale = np.log(1. + np.exp(-3.))
    self.assertAllClose(tf.convert_to_tensor(scale),
                        target_scale * np.ones(shape),
                        atol=5e-2)

    self.assertAllEqual(half_cauchy.shape, shape)
    self.assertAllGreaterEqual(half_cauchy.value, 0.)

  def testTrainableLogNormal(self):
    tf.random.set_seed(345689)
    shape = (100,)
    initializer = ed.initializers.get('trainable_log_normal')
    log_normal = initializer(shape)
    # Get distribution of rv -> get distribution of Independent.
    loc = log_normal.distribution.distribution.loc
    scale = log_normal.distribution.distribution.scale
    self.assertAllClose(tf.convert_to_tensor(loc), np.zeros(shape), atol=1e-4)
    target_scale = np.log(1. + np.exp(-3.))
    self.assertAllClose(tf.convert_to_tensor(scale),
                        target_scale * np.ones(shape),
                        atol=5e-2)

    self.assertAllGreater(tf.convert_to_tensor(log_normal), 0.)
    self.assertEqual(log_normal.shape, shape)

  def testTrainableNormal(self):
    tf.random.set_seed(345689)
    shape = (100,)
    initializer = ed.initializers.get('trainable_normal')
    normal = initializer(shape)
    # Get distribution of rv -> get distribution of Independent.
    loc = normal.distribution.distribution.loc
    scale = normal.distribution.distribution.scale
    self.assertAllClose(tf.convert_to_tensor(loc), np.zeros(shape), atol=1e-4)
    target_scale = np.log(1. + np.exp(-3.))
    self.assertAllClose(tf.convert_to_tensor(scale),
                        target_scale * np.ones(shape),
                        atol=5e-2)
    self.assertEqual(normal.shape, shape)

  def testTrainableMixtureOfDeltas(self):
    tf.random.set_seed(345689)
    shape = (100,)
    num_components = 5
    initializer = ed.initializers.TrainableMixtureOfDeltas(num_components)
    mixture_shape = list(shape) + [num_components]
    rv = initializer(shape)
    # Get distribution of rv -> get distribution of Independent.
    probs = rv.distribution.distribution.mixture_distribution.probs
    loc = rv.distribution.distribution.components_distribution.loc
    self.assertAllClose(
        probs,
        tf.broadcast_to([[1/num_components]*num_components], mixture_shape),
        atol=1e-4)
    self.assertAllClose(tf.convert_to_tensor(loc), tf.zeros_like(loc), atol=1.)
    self.assertEqual(rv.shape, shape)

  def testOrthogonalRandomFeatures(self):
    tf.random.set_seed(42)
    stddev = 0.125
    num_rows = 512
    num_cols = 2 * num_rows
    shape = (num_rows, num_cols)

    initializer = ed.initializers.OrthogonalRandomFeatures(
        stddev=stddev, random_norm=False)
    orthogonal_matrix = initializer(shape=shape)

    # Verify matrix shape.
    self.assertEqual(orthogonal_matrix.shape, tf.TensorShape(shape))

    # Verify column norm.
    squared_column_norm = num_rows * stddev**2
    norm_vector_expected = [squared_column_norm] * num_cols
    norm_vector_observed = tf.reduce_sum(orthogonal_matrix**2, axis=0)
    self.assertAllClose(norm_vector_expected, norm_vector_observed, atol=1e-5)

    # Verify orthogonal_matrix is a concatenation of two orthogonal matrices.
    inner_prod_expected = tf.eye(num_rows) * squared_column_norm

    orth_mat1 = orthogonal_matrix[:, :num_rows]
    orth_mat2 = orthogonal_matrix[:, num_rows:]
    inner_prod_observed1 = tf.matmul(orth_mat1, tf.transpose(orth_mat1))
    inner_prod_observed2 = tf.matmul(orth_mat2, tf.transpose(orth_mat2))
    self.assertAllClose(inner_prod_observed1, inner_prod_expected, atol=1e-3)
    self.assertAllClose(inner_prod_observed2, inner_prod_expected, atol=1e-3)

  def testInitializersGet(self):
    self.assertIsInstance(ed.initializers.get('trainable_normal'),
                          ed.initializers.TrainableNormal)
    # This is working correctly, but the test won't pass currently because TF
    # isn't consistent (yet).  Specifically, tf.python.keras.initializers.get('zeros')
    # returns a certain class while tf.python.keras.initializers.zeros (or Zeros)
    # currently returns v2 of that class.
    # self.assertIsInstance(ed.initializers.get('zeros'),
    #                       tf.python.keras.initializers.Zeros().__class__)


if __name__ == '__main__':
  tf.test.main()
