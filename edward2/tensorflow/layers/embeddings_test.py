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

"""Tests for Bayesian embedding layers."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


class EmbeddingTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for the Bayesian Embedding layers."""

  def setUp(self):
    self.batch_size = 5
    self.timesteps = 3
    self.input_dim = 12
    self.output_dim = 15
    self.inputs = np.random.randint(
        self.input_dim - 1, size=(self.batch_size, self.timesteps))
    super(EmbeddingTest, self).setUp()

  @parameterized.parameters(
      {"embeddings_initializer": "uniform", "all_close": True},
      {"embeddings_initializer": "trainable_normal", "all_close": False},
  )
  def testEmbedding(self, embeddings_initializer, all_close):
    layer = ed.layers.EmbeddingReparameterization(
        self.input_dim,
        output_dim=self.output_dim,
        embeddings_initializer=embeddings_initializer)
    outputs1 = tf.convert_to_tensor(layer(self.inputs))
    outputs2 = tf.convert_to_tensor(layer(self.inputs))
    self.assertEqual(outputs1.shape,
                     (self.batch_size, self.timesteps, self.output_dim))
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)

  @parameterized.parameters(
      {"embeddings_initializer": "uniform",
       "embeddings_regularizer": None,
       "all_close": True,
       "num_losses": 0},
      {"embeddings_initializer": "trainable_normal",
       "embeddings_regularizer": "normal_kl_divergence",
       "all_close": False,
       "num_losses": 1},
  )
  def testEmbeddingModel(self, embeddings_initializer, embeddings_regularizer,
                         all_close, num_losses):
    model_output_dim = 2
    model = tf.python.keras.Sequential([
        ed.layers.EmbeddingReparameterization(
            self.input_dim,
            output_dim=self.output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer),
        tf.python.keras.layers.RNN(tf.python.keras.layers.LSTMCell(5)),
        tf.python.keras.layers.Flatten(),
        tf.python.keras.layers.Dense(model_output_dim),
    ])
    outputs1 = model(self.inputs, training=True)
    outputs2 = model(self.inputs, training=True)
    self.assertEqual(outputs1.shape, (self.batch_size, model_output_dim))
    if all_close:
      self.assertAllClose(outputs1, outputs2)
    else:
      self.assertNotAllClose(outputs1, outputs2)
    self.assertLen(model.losses, num_losses)


if __name__ == "__main__":
  tf.test.main()
