# coding=utf-8
# Copyright 2022 The Edward2 Authors.
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

"""Tests for hetsngp.py."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


def test_cases():
  kernel_types = ['gaussian', 'linear']
  nums_classes = [3, 6]
  cases = []
  for kernel_type in kernel_types:
    for num_classes in nums_classes:
      cases.append({
          'testcase_name': f'_posterior_network_{kernel_type}_{num_classes}',
          'gp_kernel_type': kernel_type,
          'num_classes': num_classes
      })
  return parameterized.named_parameters(*cases)


class Classifier(tf.keras.Model):
  """Wrapper for classifiers defined below.

  Handles different architectures and differences between eager/graph execution.
  """

  def __init__(self,
               gp_kernel_type,
               num_classes,
               **kwargs):
    super().__init__()
    self.hidden_layer = tf.keras.layers.Dense(16)
    self.output_layer = ed.layers.HeteroscedasticSNGPLayer(
        num_classes=num_classes,
        num_factors=num_classes,
        gp_kernel_type=gp_kernel_type)
    self.classifier = tf.keras.Sequential(
        layers=[self.hidden_layer, self.output_layer])

  def call(self, inputs, **kwargs):
    if tf.executing_eagerly():
      return self.classifier(inputs, **kwargs)
    else:
      with tf.compat.v1.variable_scope('scope', use_resource=True):
        return self.classifier(inputs, **kwargs)


class HetSNGPTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    if not tf.executing_eagerly():
      tf.compat.v1.enable_resource_variables()
    super().setUp()

  def create_dataset(self, num_classes):
    x = np.asarray([[1.0, 2.0], [0.5, 1.5], [0.2, 0.15], [-0.3, 0.0]])
    y = np.asarray([[i % num_classes] for i in range(4)])
    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

  @test_cases()
  def test_layer_construction(self, gp_kernel_type, num_classes):
    output_layer = ed.layers.HeteroscedasticSNGPLayer(
        num_classes=num_classes,
        num_factors=num_classes,
        gp_kernel_type=gp_kernel_type)
    self.assertIsNotNone(output_layer)

  @test_cases()
  def test_model_construction(self, gp_kernel_type, num_classes):
    model = Classifier(
        num_classes=num_classes,
        gp_kernel_type=gp_kernel_type)
    self.assertIsNotNone(model)

  @test_cases()
  def test_train_step(self, gp_kernel_type, num_classes):
    x, y = self.create_dataset(num_classes)

    model = Classifier(
        num_classes=num_classes,
        gp_kernel_type=gp_kernel_type)

    out = model(x)
    self.assertIsNotNone(out)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if tf.executing_eagerly():
      optimizer = tf.keras.optimizers.Adam()
      def train_step(inputs, labels, model):
        """Defines a single training step: Update weights based on one batch."""
        with tf.GradientTape() as tape:
          log_preds = model(inputs)
          loss_value = loss_fn(labels, log_preds)

        grads = tape.gradient(loss_value, model.trainable_weights)
        self.assertIsNotNone(grads)
        grads, _ = tf.clip_by_global_norm(grads, 2.5)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

      loss_value = train_step(x, y, model).numpy()
    else:
      optimizer = tf.compat.v1.train.AdamOptimizer()
      log_preds = model(x)
      loss_value = loss_fn(y, log_preds)
      train_op = optimizer.minimize(loss_value)
      self.initialise()
      loss_value, _ = self.evaluate([loss_value, train_op])

    self.assertGreater(loss_value, 0)

  def initialise(self):
    if not tf.executing_eagerly():
      self.evaluate([tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer()])

if __name__ == '__main__':
  tf.test.main()
