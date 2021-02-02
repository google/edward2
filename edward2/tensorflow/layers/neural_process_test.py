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

"""Tests for Neural processes."""

import edward2 as ed
import numpy as np
import tensorflow as tf


def train_neural_process(model,
                         train_data,
                         valid_data,
                         num_epochs,
                         batch_size,
                         learning_rate=1e-4):
  """Trains the NeuralProcess model.

  Validation data is used for early stopping,

  Args:
    model: A NeuralProcess Model subclassing Keras model.
    train_data: (4-tuple of tensors) Values of x and y for contexts and targets.
    valid_data: 4-tuple of tensors) Values of x and y for contexts and targets.
    num_epochs: (int) Number of epochs to train the model for.
    batch_size: (int) Size of batch.
    learning_rate: (float) Learning rate for Adam optimizer.

  Returns:
    best_loss: (float) Average validation loss of best early-stopped model.
  """
  optimizer = tf.python.keras.optimizers.Adam(learning_rate)
  context_x, context_y, target_x, target_y = train_data
  valid_context_x, valid_context_y, valid_target_x, valid_target_y = valid_data
  train_data_size = target_x.shape[0]
  num_updates_per_epoch = train_data_size//batch_size
  best_loss = np.inf
  valid_query = (valid_context_x, valid_context_y), valid_target_x

  for _ in range(num_epochs):
    for i in range(num_updates_per_epoch):
      start_idx, end_idx = batch_size*i, batch_size*(i+1)
      batch_query = ((context_x[start_idx:end_idx],
                      context_y[start_idx:end_idx]),
                     target_x[start_idx:end_idx])
      batch_target_y = target_y[start_idx:end_idx]
      num_targets = tf.shape(batch_target_y)[1]
      with tf.GradientTape() as tape:
        predictive_dist = model(batch_query, batch_target_y)
        log_p = predictive_dist.log_prob(batch_target_y)
        kl = tf.tile(model.losses[-1], [1, num_targets])
        loss = -tf.reduce_mean(log_p - kl/tf.cast(num_targets, tf.float32))
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    predictive_dist = model(valid_query, valid_target_y)
    log_p = predictive_dist.log_prob(valid_target_y)
    kl = tf.tile(model.losses[-1], [1, tf.shape(valid_target_y)[1]])
    valid_loss = -tf.reduce_mean(log_p - kl/tf.cast(num_targets, tf.float32))
    if valid_loss < best_loss:
      best_loss = valid_loss

  return best_loss


class NeuralProcessTest(tf.test.TestCase):

  def setUp(self):
    # Create a dummy multi-task fake dataset
    num_train_problems = 32
    num_valid_problems = 32
    num_targets = 50
    num_contexts = 10
    input_dim = 5

    def _create_fake_dataset(num_problems):
      target_x = np.random.rand(num_problems,
                                num_targets,
                                input_dim).astype(np.float32)
      target_y = np.random.rand(num_problems,
                                num_targets,
                                1).astype(np.float32)
      context_x, context_y = (target_x[:, :num_contexts, :],
                              target_y[:, :num_contexts, :])
      return (context_x, context_y, target_x, target_y)

    self.train_data = _create_fake_dataset(num_train_problems)
    self.valid_data = _create_fake_dataset(num_valid_problems)

    hidden_size = 128
    num_latents = 16

    np_attention_wrapper = ed.layers.Attention(
        rep='identity', output_sizes=None, att_type='uniform')
    self.np_model = ed.layers.NeuralProcess(
        latent_encoder_sizes=[hidden_size]*4,
        num_latents=num_latents,
        decoder_sizes=[hidden_size]*2 + [2],
        use_deterministic_path=True,
        deterministic_encoder_sizes=[hidden_size]*4,
        attention_wrapper=np_attention_wrapper)

    anp_attention_wrapper = ed.layers.Attention(
        rep='mlp', output_sizes=[hidden_size]*2, att_type='multihead')
    self.anp_model = ed.layers.NeuralProcess(
        latent_encoder_sizes=[hidden_size]*4,
        num_latents=num_latents,
        decoder_sizes=[hidden_size]*2 + [2],
        use_deterministic_path=True,
        deterministic_encoder_sizes=[hidden_size]*4,
        attention_wrapper=anp_attention_wrapper)

    self.models = [self.np_model, self.anp_model]
    self.num_latents, self.hidden_size, self.num_targets = (num_latents,
                                                            hidden_size,
                                                            num_targets)
    super(NeuralProcessTest, self).setUp()

  def testTermination(self):
    for model in self.models:
      validation_loss = train_neural_process(
          model,
          self.train_data,
          self.valid_data,
          num_epochs=2,
          batch_size=16,
          learning_rate=1e-4)

      self.assertGreaterEqual(validation_loss, 0.)

  def testLatentEncoder(self):
    valid_context_x, valid_context_y, _, _ = self.valid_data
    batch_size = valid_context_x.shape[0]

    for model in self.models:
      dist = model.latent_encoder(valid_context_x, valid_context_y).distribution
      self.assertEqual(dist.loc.shape, (batch_size, self.num_latents))
      self.assertEqual(dist.scale.shape,
                       (batch_size, self.num_latents, self.num_latents))

  def testDeterministicEncoder(self):
    valid_context_x, valid_context_y, valid_target_x, _ = self.valid_data
    batch_size = valid_context_x.shape[0]

    for model in self.models:
      embedding = model.deterministic_encoder(
          valid_context_x, valid_context_y, valid_target_x)
      self.assertEqual(embedding.shape, (batch_size, self.num_targets,
                                         self.hidden_size))

  def testCall(self):
    (valid_context_x,
     valid_context_y,
     valid_target_x,
     valid_target_y) = self.valid_data
    batch_size = valid_context_x.shape[0]

    for model in self.models:
      query = (valid_context_x, valid_context_y), valid_target_x
      # test 'training' when target_y is available
      predictive_dist = model(query, valid_target_y)
      self.assertEqual(predictive_dist.loc.shape, (batch_size, self.num_targets,
                                                   1))
      self.assertEqual(predictive_dist.scale.shape,
                       (batch_size, self.num_targets, 1, 1))
      self.assertAllGreaterEqual(model.losses, 0.)

      # test 'testing' when target_y is unavailable
      predictive_dist = model(query)
      self.assertEqual(predictive_dist.loc.shape, (batch_size, self.num_targets,
                                                   1))
      self.assertEqual(predictive_dist.scale.shape,
                       (batch_size, self.num_targets, 1, 1))


if __name__ == '__main__':
  tf.test.main()
