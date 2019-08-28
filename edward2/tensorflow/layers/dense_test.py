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

"""Tests for Bayesian dense layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf

tfe = tf1.contrib.eager


class DenseTest(parameterized.TestCase, tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testTrainableNormalStddevConstraint(self):
    layer = ed.layers.DenseReparameterization(
        100, kernel_initializer="trainable_normal")
    inputs = tf.random.normal([1, 1])
    out = layer(inputs)
    stddev = layer.kernel.distribution.stddev()
    self.evaluate(tf1.global_variables_initializer())
    res, _ = self.evaluate([stddev, out])
    self.assertAllGreater(res, 0.)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseDVI,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseFlipout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseReparameterization,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "zeros",
       "all_close": True},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "trainable_normal",
       "bias_initializer": "zeros",
       "all_close": False},
      {"layer": ed.layers.DenseVariationalDropout,
       "kernel_initializer": "zeros",
       "bias_initializer": "trainable_normal",
       "all_close": False},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testDenseKernel(self,
                      layer,
                      kernel_initializer,
                      bias_initializer,
                      all_close):
    tf.keras.backend.set_learning_phase(1)  # training time
    inputs = np.random.rand(5, 3, 12).astype(np.float32)
    model = layer(4,
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer,
                  activation=tf.nn.relu)
    outputs1 = model(inputs)
    outputs2 = model(inputs)
    self.evaluate(tf1.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    if layer != ed.layers.DenseDVI:
      self.assertAllGreaterEqual(res1, 0.)
    if all_close:
      self.assertAllClose(res1, res2)
    else:
      self.assertNotAllClose(res1, res2)
    model.get_config()

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testDenseMean(self, layer):
    """Tests that forward pass can use other values, e.g., posterior mean."""
    tf.keras.backend.set_learning_phase(0)  # test time
    def take_mean(f, *args, **kwargs):
      """Sets random variable value to its mean."""
      rv = f(*args, **kwargs)
      rv._value = rv.distribution.mean()
      return rv
    inputs = np.random.rand(5, 3, 7).astype(np.float32)
    model = layer(4, activation=tf.nn.relu, use_bias=False)
    outputs1 = model(inputs)
    with ed.trace(take_mean):
      outputs2 = model(inputs)
    self.evaluate(tf1.global_variables_initializer())
    res1, res2 = self.evaluate([outputs1, outputs2])
    self.assertEqual(res1.shape, (5, 3, 4))
    self.assertNotAllClose(res1, res2)
    if layer != ed.layers.DenseDVI:
      self.assertAllClose(res2, np.zeros((5, 3, 4)), atol=1e-4)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testDenseLoss(self, layer):
    tf.keras.backend.set_learning_phase(1)  # training time
    features = np.random.rand(5, 12).astype(np.float32)
    labels = np.random.rand(5, 10).astype(np.float32)
    model = layer(10)

    # Imagine this is the 1st epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions = model(features)  # first call forces build
      model(features)  # ensure robustness after multiple calls
      nll = tf.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      self.assertIn(v, model.variables)

    # This will be fine, since the layer was built inside this tape, and thus
    # the distribution init ops were inside this tape.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

    # Imagine this is the 2nd epoch.
    with tf.GradientTape(persistent=True) as tape:
      predictions = model(features)  # build is not called
      nll = tf.keras.losses.mean_squared_error(labels, predictions)
      kl = sum(model.losses)

    variables = [model.kernel_initializer.mean, model.kernel_initializer.stddev]
    for v in variables:
      self.assertIn(v, model.variables)

    # This would fail, since the layer was built inside the tape from the 1st
    # epoch, and thus the distribution init ops were inside that tape instead of
    # this tape. By using a callable for the variable, this will no longer fail.
    grads = tape.gradient(nll, variables)
    for grad in grads:
      self.assertIsNotNone(grad)
    grads = tape.gradient(kl, variables)
    for grad in grads:
      self.assertIsNotNone(grad)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testDenseModel(self, layer):
    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        layer(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.evaluate(tf1.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    if layer == ed.layers.DenseHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @parameterized.parameters(
      {"layer": ed.layers.DenseDVI},
      {"layer": ed.layers.DenseFlipout},
      {"layer": ed.layers.DenseReparameterization},
      {"layer": ed.layers.DenseVariationalDropout},
      {"layer": ed.layers.DenseHierarchical},
  )
  @tfe.run_test_in_graph_and_eager_modes
  def testDenseSubclass(self, layer):
    class DenseSubclass(layer):
      pass

    inputs = np.random.rand(3, 4, 4, 1).astype(np.float32)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(3,
                               kernel_size=2,
                               padding="SAME",
                               activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        DenseSubclass(2, activation=None),
    ])
    outputs = model(inputs, training=True)
    self.evaluate(tf1.global_variables_initializer())
    res = self.evaluate(outputs)
    self.assertEqual(res.shape, (3, 2))
    if layer == ed.layers.DenseHierarchical:
      self.assertLen(model.losses, 3)
    else:
      self.assertLen(model.losses, 1)

  @tfe.run_test_in_graph_and_eager_modes
  def testDenseDVIIsDeterministic(self):
    """Tests that DenseDVI network has a deterministic loss function."""
    features = np.random.rand(3, 2).astype(np.float32)
    labels = np.random.rand(3, 1).astype(np.float32)
    model = tf.keras.Sequential([
        ed.layers.DenseDVI(5, activation=tf.nn.relu),
        ed.layers.DenseDVI(1, activation=None),
    ])
    outputs = model(features, training=True)
    nll = -tf.reduce_sum(outputs.distribution.log_prob(labels))
    kl = sum(model.losses)
    loss = nll + kl
    self.evaluate(tf1.global_variables_initializer())
    res1 = self.evaluate(loss)
    res2 = self.evaluate(loss)
    self.assertEqual(res1, res2)

  @tfe.run_test_in_graph_and_eager_modes
  def testDenseDVIMoments(self):
    """Verifies DenseDVI's moments empirically with samples."""
    tf.random.set_seed(377269)
    batch_size = 3
    num_features = 5
    units = 128
    num_samples = 50000
    inputs = tf.cast(np.random.rand(batch_size, num_features), dtype=tf.float32)
    layer = ed.layers.DenseDVI(units, activation=tf.nn.relu)

    outputs1 = layer(inputs)
    mean1 = outputs1.distribution.mean()
    covariance1 = outputs1.distribution.covariance()

    kernel_samples = layer.kernel.distribution.sample(num_samples)
    outputs2 = layer.activation(
        tf.einsum("bd,sdu->sbu", inputs, kernel_samples) +
        tf.reshape(layer.bias, [1, 1, units]))
    mean2 = tf.reduce_mean(outputs2, axis=0)
    centered_outputs2 = tf.transpose(a=outputs2 - mean2, perm=[1, 2, 0])
    covariance2 = tf.matmul(centered_outputs2,
                            centered_outputs2,
                            transpose_b=True) / float(num_samples)

    self.evaluate(tf1.global_variables_initializer())
    mean1_val, covariance1_val, mean2_val, covariance2_val = self.evaluate(
        [mean1, covariance1, mean2, covariance2])
    # Check % of mismatches is not too high according to heuristic thresholds.
    num_mismatches = np.sum(np.abs(mean1_val - mean2_val) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units)
    self.assertLessEqual(percent_mismatches, 0.05)
    num_mismatches = np.sum(np.abs(covariance1_val - covariance2_val) > 5e-3)
    percent_mismatches = num_mismatches / float(batch_size * units * units)
    self.assertLessEqual(percent_mismatches, 0.05)


if __name__ == "__main__":
  tf.test.main()
