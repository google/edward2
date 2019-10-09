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

# Lint as: python3
"""Thompson sampling for contextual bandit problems via offline GNPs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.experimental.attentive_uncertainty import generalized_neural_process
import tensorflow as tf

from deep_contextual_bandits import bandit_algorithm  # local file import
from deep_contextual_bandits import contextual_dataset  # local file import


class OfflineContextualBandits(bandit_algorithm.BanditAlgorithm):
  """Thompson sampling via offline neural processes."""

  def __init__(self,
               name,
               hparams):
    self.name = name
    self.hparams = hparams
    self.verbose = getattr(hparams, 'verbose', True)

    self.t = 0
    self.data_h = contextual_dataset.ContextualDataset(
        hparams.context_dim, hparams.num_actions, intercept=False)

    if self.verbose:
      print('Initializing model {}.'.format(self.name))
    self.gnp = generalized_neural_process.Regressor(
        input_dim=hparams.context_dim,
        output_dim=hparams.num_actions,
        x_encoder_net_sizes=hparams.x_encoder_net_sizes,
        x_y_encoder_net_sizes=hparams.x_y_encoder_net_sizes,
        global_latent_net_sizes=hparams.global_latent_net_sizes,
        local_latent_net_sizes=hparams.local_latent_net_sizes,
        decoder_net_sizes=hparams.decoder_net_sizes,
        heteroskedastic_net_sizes=hparams.heteroskedastic_net_sizes,
        att_type=hparams.att_type,
        att_heads=hparams.att_heads,
        model_type=hparams.model_type,
        activation=hparams.activation,
        output_activation=hparams.output_activation,
        data_uncertainty=hparams.data_uncertainty,
        beta=hparams.beta,
        temperature=hparams.temperature,
        model_path=hparams.model_path)

  def action(self, context):
    """Samples rewards from posterior, and chooses best action accordingly.

    Args:
      context: A d-dimensional np.ndarray with the context.

    Returns:
      Greedy action based on Thompson sampling.
    """
    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    vals = []

    context = tf.to_float(context)
    contexts, rewards, actions = self.data_h.get_data_with_weights()
    historical_x = tf.to_float(tf.expand_dims(contexts, axis=0))
    historical_y = tf.to_float(tf.expand_dims(rewards*actions, axis=0))
    target_x = tf.expand_dims(tf.reshape(context, [1, -1]), axis=0)
    target_y = None

    predictions = self.gnp(historical_x, historical_y, target_x, target_y)
    vals = tf.squeeze(predictions.distribution.mean())

    return tf.argmax(vals).numpy()

  def update(self, context, action, reward):
    """Updates the posterior of the SNP model.

    For an offline SNP model, the posterior gets directly updated by
    updating the observed dataset. No parameter updates are needed.

    Args:
      context: A d-dimensional np.ndarray with the context.
      action: Integer between 0 and k-1 representing the chosen arm.
      reward: Real number representing the reward for the (context, action).

    Returns:
      None.
    """
    self.t += 1
    self.data_h.add(context, action, reward)
    if self.verbose and self.t % 100 == 0:
      print('Number of contexts=', self.t)


