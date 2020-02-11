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

"""Tests for discrete flows."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf


class DiscreteFlowsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testDiscreteAutoregressiveFlowCall(self, loc_only):
    batch_size = 3
    vocab_size = 79
    length = 5
    if loc_only:
      units = vocab_size
      network = ed.layers.MADE(units, [])
    else:
      units = 2 * vocab_size
      mask = tf.reshape([0] * vocab_size + [-1e10] + [0] * (vocab_size - 1),
                        [1, 1, 2 * vocab_size])
      network_ = ed.layers.MADE(units, [])
      network = lambda inputs, **kwargs: mask + network_(inputs, **kwargs)
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.DiscreteAutoregressiveFlow(network, 1.)
    outputs = layer(inputs)
    self.assertEqual(outputs.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs, 0)
    self.assertAllLessEqual(outputs, vocab_size - 1)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testDiscreteAutoregressiveFlowSample(self, loc_only):
    batch_size = 5
    length = 2
    vocab_size = 2
    if loc_only:
      units = vocab_size
      network = ed.layers.MADE(units, [])
    else:
      units = 2 * vocab_size
      mask = tf.reshape([0] * vocab_size + [-1e10] + [0] * (vocab_size - 1),
                        [1, 1, 2 * vocab_size])
      network_ = ed.layers.MADE(units, [])
      network = lambda inputs, **kwargs: mask + network_(inputs, **kwargs)
    layer = ed.layers.DiscreteAutoregressiveFlow(network, 1.)
    logits = tf.tile(tf.random.normal([length, vocab_size])[tf.newaxis],
                     [batch_size, 1, 1])
    base = ed.OneHotCategorical(logits=logits, dtype=tf.float32)
    outputs = layer(base)
    self.assertEqual(outputs.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(tf.convert_to_tensor(outputs), 0)
    self.assertAllLessEqual(tf.convert_to_tensor(outputs), vocab_size - 1)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testDiscreteAutoregressiveFlowInverse(self, loc_only):
    batch_size = 2
    vocab_size = 79
    length = 5
    if loc_only:
      units = vocab_size
      network = ed.layers.MADE(units, [])
    else:
      units = 2 * vocab_size
      mask = tf.reshape([0] * vocab_size + [-1e10] + [0] * (vocab_size - 1),
                        [1, 1, 2 * vocab_size])
      network_ = ed.layers.MADE(units, [])
      network = lambda inputs, **kwargs: mask + network_(inputs, **kwargs)
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.DiscreteAutoregressiveFlow(network, 1.)
    rev_fwd_inputs = layer.reverse(layer(inputs))
    fwd_rev_inputs = layer(layer.reverse(inputs))
    self.assertAllClose(inputs, rev_fwd_inputs, rtol=1e-4, atol=1e-4)
    self.assertAllClose(inputs, fwd_rev_inputs, rtol=1e-4, atol=1e-4)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testDiscreteAutoregressiveFlowRandomVariable(self, loc_only):
    batch_size = 2
    length = 4
    vocab_size = 5
    if loc_only:
      units = vocab_size
      network = ed.layers.MADE(units, [])
    else:
      units = 2 * vocab_size
      mask = tf.reshape([0] * vocab_size + [-1e10] + [0] * (vocab_size - 1),
                        [1, 1, 2 * vocab_size])
      network_ = ed.layers.MADE(units, [])
      network = lambda inputs, **kwargs: mask + network_(inputs, **kwargs)
    base = ed.OneHotCategorical(logits=tf.random.normal([batch_size,
                                                         length,
                                                         vocab_size]),
                                dtype=tf.float32)
    flow = ed.layers.DiscreteAutoregressiveFlow(network, 1.)
    flow_rv = flow(base)
    self.assertEqual(flow_rv.dtype, tf.float32)

    self.assertEqual(flow_rv.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(tf.convert_to_tensor(flow_rv), 0)
    self.assertAllLessEqual(tf.convert_to_tensor(flow_rv), vocab_size - 1)

    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    outputs = flow(inputs)
    rev_outputs = flow.reverse(outputs)
    self.assertAllClose(inputs, rev_outputs)

    inputs_log_prob = base.distribution.log_prob(inputs)
    outputs_log_prob = flow_rv.distribution.log_prob(outputs)
    self.assertEqual(inputs_log_prob.shape, (batch_size, length))
    self.assertAllClose(inputs_log_prob, outputs_log_prob)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def testDiscreteAutoregressiveFlowReverseGradients(self, loc_only):
    batch_size = 2
    length = 4
    vocab_size = 2
    if loc_only:
      units = vocab_size
      network_ = ed.layers.MADE(units, [16, 16])
      network = network_
    else:
      units = 2 * vocab_size
      network_ = ed.layers.MADE(units, [16, 16])
      mask = tf.reshape([0] * vocab_size + [-1e10] + [0] * (vocab_size - 1),
                        [1, 1, 2 * vocab_size])
      network = lambda inputs, **kwargs: mask + network_(inputs, **kwargs)
    with tf.GradientTape() as tape:
      base = ed.OneHotCategorical(
          logits=tf.random.normal([batch_size, length, vocab_size]))
      flow = ed.layers.DiscreteAutoregressiveFlow(network, 1.)
      flow_rv = flow(base)
      features = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
      features = tf.one_hot(features, depth=vocab_size, dtype=tf.float32)
      loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
          labels=flow.reverse(features),
          logits=flow_rv.distribution.base.logits))
    grads = tape.gradient(loss, network_.weights)
    for grad in grads:
      self.assertIsNotNone(grad)

  def testDiscreteBipartiteFlowCall(self):
    batch_size = 3
    vocab_size = 79
    length = 5
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.DiscreteBipartiteFlow(
        lambda inputs, **kwargs: tf.identity(inputs),
        mask=tf.random.uniform([length], minval=0, maxval=2, dtype=tf.int32),
        temperature=1.)
    outputs = layer(inputs)
    self.assertEqual(outputs.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs, 0)
    self.assertAllLessEqual(outputs, vocab_size - 1)

  def testDiscreteBipartiteFlowInverse(self):
    batch_size = 2
    vocab_size = 79
    length = 5
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.DiscreteBipartiteFlow(
        lambda inputs, **kwargs: tf.identity(inputs),
        mask=tf.random.uniform([length], minval=0, maxval=2, dtype=tf.int32),
        temperature=1.)
    rev_fwd_inputs = layer.reverse(layer(inputs))
    fwd_rev_inputs = layer(layer.reverse(inputs))
    self.assertAllClose(inputs, rev_fwd_inputs)
    self.assertAllClose(inputs, fwd_rev_inputs)

  def testSinkhornAutoregressiveFlowCall(self):
    batch_size = 3
    vocab_size = 79
    length = 5
    units = vocab_size ** 2
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.SinkhornAutoregressiveFlow(
        ed.layers.MADE(units, []), 1.)
    outputs = layer(inputs)
    self.assertEqual(outputs.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs, 0)
    self.assertAllLessEqual(outputs, vocab_size - 1)

  def testDiscreteSinkhornFlowInverse(self):
    batch_size = 2
    vocab_size = 79
    length = 5
    units = vocab_size ** 2
    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    layer = ed.layers.SinkhornAutoregressiveFlow(
        ed.layers.MADE(units, []), 1.)
    rev_fwd_inputs = layer.reverse(layer(inputs))
    fwd_rev_inputs = layer(layer.reverse(inputs))
    self.assertAllEqual(inputs, rev_fwd_inputs)
    self.assertAllEqual(inputs, fwd_rev_inputs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
