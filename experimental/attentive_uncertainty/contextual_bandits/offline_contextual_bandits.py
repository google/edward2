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

# Lint as: python3
"""Thompson sampling for contextual bandit problems via offline SNPs.
"""

from experimental.attentive_uncertainty import regressor  # local file import
from experimental.attentive_uncertainty.contextual_bandits import utils  # local file import
import numpy as np
import tensorflow.compat.v1 as tf

from deep_contextual_bandits import bandit_algorithm  # local file import
from deep_contextual_bandits import contextual_dataset  # local file import


class OfflineContextualBandits(bandit_algorithm.BanditAlgorithm):
  """Thompson sampling via offline strutured neural processes."""

  def __init__(self,
               name,
               hparams):
    self.name = name
    self.hparams = hparams
    self.verbose = getattr(hparams, 'verbose', True)
    self._is_anp = getattr(hparams, 'is_anp', False)
    if self._is_anp:
      input_dim = hparams.context_dim
      output_dim = hparams.num_actions
    else:
      input_dim = hparams.context_dim + hparams.num_actions
      output_dim = 1

    self.t = 0
    self.data_h = contextual_dataset.ContextualDataset(
        hparams.context_dim, hparams.num_actions, intercept=False)

    if self.verbose:
      print('Initializing model {}.'.format(self.name))
    self.snp = regressor.Regressor(
        input_dim=input_dim,
        output_dim=output_dim,
        x_encoder_sizes=hparams.x_encoder_sizes,
        x_y_encoder_sizes=hparams.x_y_encoder_sizes,
        global_latent_net_sizes=hparams.global_latent_net_sizes,
        local_latent_net_sizes=hparams.local_latent_net_sizes,
        heteroskedastic_net_sizes=hparams.heteroskedastic_net_sizes,
        att_type=hparams.att_type,
        att_heads=hparams.att_heads,
        uncertainty_type=hparams.uncertainty_type,
        mean_att_type=hparams.mean_att_type,
        scale_att_type_1=hparams.scale_att_type_1,
        scale_att_type_2=hparams.scale_att_type_2,
        activation=hparams.activation,
        output_activation=hparams.output_activation,
        data_uncertainty=hparams.data_uncertainty,
        local_variational=hparams.local_variational,
        model_path=hparams.model_path)

    self._one_hot_vectors = tf.one_hot(
        indices=np.arange(hparams.num_actions),
        depth=hparams.num_actions)

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
    if self._is_anp:
      contexts, rewards, actions = self.data_h.get_data_with_weights()
      historical_x = tf.to_float(tf.expand_dims(contexts, axis=0))
      historical_y = tf.to_float(tf.expand_dims(rewards*actions, axis=0))
      target_x = tf.expand_dims(tf.reshape(context, [1, -1]), axis=0)
    else:
      contexts, rewards, actions = utils.get_data_with_masked_rewards(
          self.data_h)
      context_action_pairs = tf.concat([contexts, actions], axis=-1)

      historical_x = tf.to_float(tf.expand_dims(context_action_pairs, axis=0))
      historical_y = tf.to_float(rewards.reshape(1, -1, 1))
      tiled_context = tf.concat(
          [tf.tile(tf.reshape(context, [1, -1]), [self.hparams.num_actions, 1]),
           self._one_hot_vectors], axis=-1
      )
      target_x = tf.expand_dims(tiled_context, axis=0)
    target_y = None

    predictions = self.snp(historical_x, historical_y, target_x, target_y)
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

