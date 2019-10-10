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

"""Tests for discrete flows."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
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
    self.evaluate(tf1.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertEqual(outputs_val.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs_val, 0)
    self.assertAllLessEqual(outputs_val, vocab_size - 1)

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
    _ = outputs.value  # need to do this to instantiate tf.variables
    self.evaluate(tf1.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(res, 0)
    self.assertAllLessEqual(res, vocab_size - 1)

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
    self.evaluate(tf1.global_variables_initializer())
    inputs_val, rev_fwd_inputs_val, fwd_rev_inputs_val = self.evaluate(
        [inputs, rev_fwd_inputs, fwd_rev_inputs])
    self.assertAllClose(inputs_val, rev_fwd_inputs_val, rtol=1e-4, atol=1e-4)
    self.assertAllClose(inputs_val, fwd_rev_inputs_val, rtol=1e-4, atol=1e-4)

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

    self.evaluate(tf1.global_variables_initializer())
    res = self.evaluate(flow_rv)
    self.assertEqual(res.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(res, 0)
    self.assertAllLessEqual(res, vocab_size - 1)

    inputs = np.random.randint(0, vocab_size - 1, size=(batch_size, length))
    inputs = tf.one_hot(inputs, depth=vocab_size, dtype=tf.float32)
    outputs = flow(inputs)
    rev_outputs = flow.reverse(outputs)
    inputs_val, rev_outputs_val = self.evaluate([inputs, rev_outputs])
    self.assertAllClose(inputs_val, rev_outputs_val)

    inputs_log_prob = base.distribution.log_prob(inputs)
    outputs_log_prob = flow_rv.distribution.log_prob(outputs)
    res1, res2 = self.evaluate([inputs_log_prob, outputs_log_prob])
    self.assertEqual(res1.shape, (batch_size, length))
    self.assertAllClose(res1, res2)

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
    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(grads)
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
    self.evaluate(tf1.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertEqual(outputs_val.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs_val, 0)
    self.assertAllLessEqual(outputs_val, vocab_size - 1)

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
    self.evaluate(tf1.global_variables_initializer())
    inputs_val, rev_fwd_inputs_val, fwd_rev_inputs_val = self.evaluate(
        [inputs, rev_fwd_inputs, fwd_rev_inputs])
    self.assertAllClose(inputs_val, rev_fwd_inputs_val)
    self.assertAllClose(inputs_val, fwd_rev_inputs_val)

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
    self.evaluate(tf1.global_variables_initializer())
    outputs_val = self.evaluate(outputs)
    self.assertEqual(outputs_val.shape, (batch_size, length, vocab_size))
    self.assertAllGreaterEqual(outputs_val, 0)
    self.assertAllLessEqual(outputs_val, vocab_size - 1)

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
    self.evaluate(tf1.global_variables_initializer())
    inputs_val, rev_fwd_inputs_val, fwd_rev_inputs_val = self.evaluate(
        [inputs, rev_fwd_inputs, fwd_rev_inputs])
    self.assertAllEqual(inputs_val, rev_fwd_inputs_val)
    self.assertAllEqual(inputs_val, fwd_rev_inputs_val)


if __name__ == '__main__':
  tf.test.main()
