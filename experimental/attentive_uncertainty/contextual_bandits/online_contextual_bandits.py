# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Thompson sampling for contextual bandit problems via online learning of SNPs.
"""

from experimental.attentive_uncertainty import regressor  # local file import
from experimental.attentive_uncertainty import utils  # local file import
from experimental.attentive_uncertainty.contextual_bandits import utils as bandit_utils  # local file import
import numpy as np
import tensorflow.compat.v1 as tf

from deep_contextual_bandits import bandit_algorithm  # local file import
from deep_contextual_bandits import contextual_dataset  # local file import


class OnlineContextualBandits(bandit_algorithm.BanditAlgorithm):
  """Thompson sampling via online structured neural processes."""

  def __init__(self,
               name,
               hparams,
               optimizer='RMS'):
    self.name = name
    self.hparams = hparams
    self.verbose = getattr(hparams, 'verbose', True)

    self.update_freq_lr = hparams.training_freq
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.num_epochs = hparams.training_epochs
    self.data_h = contextual_dataset.ContextualDataset(
        hparams.context_dim, hparams.num_actions, intercept=False)

    self.gradient_updates = tf.Variable(0, trainable=False)
    if self.hparams.activate_decay:
      self.lr = tf.train.inverse_time_decay(
          self.hparams.initial_lr, self.gradient_updates,
          1, self.hparams.lr_decay_rate)
    else:
      self.lr = tf.Variable(self.hparams.initial_lr, trainable=False)
    optimizer = tf.train.RMSPropOptimizer(self.lr)
    self._optimizer_config = {'optimizer': optimizer,
                              'max_grad_norm': hparams.max_grad_norm}

    if self.verbose:
      print('Initializing model {}.'.format(self.name))
    self.snp = regressor.Regressor(
        input_dim=hparams.context_dim + hparams.num_actions,
        output_dim=1,
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

    self._step = tf.function(utils.mse_step.python_function)  # pytype: disable=module-attr

    self._one_hot_vectors = tf.one_hot(
        indices=np.arange(hparams.num_actions),
        depth=hparams.num_actions)

  def action(self, context):
    """Samples rewards from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    vals = []
    states, rewards, actions = bandit_utils.get_data_with_masked_rewards(
        self.data_h)
    state_action_pairs = tf.concat([states, actions], axis=-1)

    historical_x = tf.to_float(tf.expand_dims(state_action_pairs, axis=0))
    historical_y = tf.to_float(rewards.reshape(1, -1, 1))

    context = tf.to_float(context)
    tiled_context = tf.concat(
        [tf.tile(tf.reshape(context, [1, -1]), [self.hparams.num_actions, 1]),
         self._one_hot_vectors], axis=-1
    )
    target_x = tf.expand_dims(tiled_context, axis=0)
    target_y = None

    prediction = self.snp(historical_x, historical_y, target_x, target_y)
    vals = tf.squeeze(prediction.distribution.mean())

    return tf.argmax(vals).numpy()

  def update(self, context, action, reward):
    """Updates the posterior."""

    self.t += 1
    self.data_h.add(context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:
      print('Number of contexts observed=', self.t)
      if self.hparams.reset_lr:
        self.lr = self.hparams.initial_lr
      self.train(self.data_h, self.num_epochs)

  def train(self, data, num_steps):
    """Trains the network for num_steps, using the provided data.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.
    """

    if self.verbose:
      print('Training {} for {} steps...'.format(self.name, num_steps))

    avg_nll, avg_mse, avg_local_z_kl, avg_global_z_kl = 0., 0., 0., 0.
    for _ in range(num_steps):
      states, rewards, actions = bandit_utils.get_batch_with_masked_rewards(
          self.data_h,
          self.hparams.batch_size)
      state_action_pairs = tf.concat([states, actions], axis=-1)
      target_x = tf.to_float(tf.expand_dims(state_action_pairs, axis=0))
      target_y = tf.to_float(rewards.reshape(1, -1, 1))
      num_historical = tf.random.uniform(
          (), 1, max(2, target_x.shape[1]), dtype=tf.dtypes.int32)
      historical_x = target_x[:, :num_historical]
      historical_y = target_y[:, :num_historical]
      unseen_targets = target_y[:, num_historical:]
      data = (historical_x, historical_y, target_x, target_y, unseen_targets)

      nll, mse, local_z_kl, global_z_kl = self._step(self.snp,
                                                     data,
                                                     self._optimizer_config)
      avg_nll += nll
      avg_mse += mse
      avg_local_z_kl += local_z_kl
      avg_global_z_kl += global_z_kl
      self.gradient_updates.assign_add(1)

    if self.verbose:
      avg_nll /= num_steps
      avg_mse /= num_steps
      avg_local_z_kl /= num_steps
      avg_global_z_kl /= num_steps
      print('Average nll: {}, mse: {}, local kl: {} global kl: {}'
            .format(avg_nll, avg_mse, avg_local_z_kl, avg_global_z_kl))
