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

"""Pretrains structured neural processes for the wheel bandit task.
"""

from experimental.attentive_uncertainty import regressor  # local file import
from experimental.attentive_uncertainty import utils  # local file import
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.enable_eager_execution()


def sample_single_wheel_bandit_data(num_datapoints,
                                    num_actions,
                                    context_dim,
                                    delta,
                                    mean_v,
                                    std_v,
                                    mu_large,
                                    std_large):
  """Samples from Wheel bandit game (see Riquelme et al. (2018)).

  Args:
    num_datapoints: Number (n) of (context, action, reward) triplets to sample.
    num_actions: (a) Number of actions.
    context_dim: (c) Number of dimensions in the context.
    delta: Exploration parameter: high reward in one region if norm above delta.
    mean_v: Mean reward for each action if context norm is below delta.
    std_v: Gaussian reward std for each action if context norm is below delta.
    mu_large: Mean reward for optimal action if context norm is above delta.
    std_large: Reward std for optimal action if context norm is above delta.

  Returns:
    context_action_pairs: Sampled (context, action) matrix of size (n, c + a).
    rewards: Sampled reward matrix of size (n, 1).
  """
  data = []
  actions = []
  rewards = []

  # Sample uniform contexts in unit ball.
  while len(data) < num_datapoints:
    raw_data = np.random.uniform(-1, 1, (int(num_datapoints / 3), context_dim))

    for i in range(raw_data.shape[0]):
      if np.linalg.norm(raw_data[i, :]) <= 1:
        data.append(raw_data[i, :])

  contexts = np.stack(data)[:num_datapoints, :]

  # Sample rewards and random actions.
  for i in range(num_datapoints):
    r = [np.random.normal(mean_v[j], std_v[j]) for j in range(num_actions)]
    if np.linalg.norm(contexts[i, :]) >= delta:
      # Large reward in the right region for the context.
      r_big = np.random.normal(mu_large, std_large)
      if contexts[i, 0] > 0:
        if contexts[i, 1] > 0:
          r[0] = r_big
        else:
          r[1] = r_big
      else:
        if contexts[i, 1] > 0:
          r[2] = r_big
        else:
          r[3] = r_big
    one_hot_vector = np.zeros((5))
    random_action = np.random.randint(num_actions)
    one_hot_vector[random_action] = 1
    actions.append(one_hot_vector)
    rewards.append(r[random_action])

  rewards = np.expand_dims(np.array(rewards), -1)
  context_action_pairs = np.hstack([contexts, actions])
  perm = np.random.permutation(len(rewards))
  return context_action_pairs[perm, :], rewards[perm, :]


def get_single_wheel_data(num_datapoints, num_actions, context_dim, delta):
  """Samples data for a single wheel with default benchmark configuration.

  Args:
    num_datapoints: Number (n) of (context, action, reward) triplets to sample.
    num_actions: (a) Number of actions.
    context_dim: (c) Number of dimensions in the context.
    delta: Exploration parameter: high reward in one region if norm above delta.

  Returns:
    context_action_pairs: Sampled (context, action) matrix of size (n, c + a).
    rewards: Sampled reward matrix of size (n, 1).
  """
  mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
  std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
  mu_large = 50
  std_large = 0.01
  context_action_pairs, rewards = sample_single_wheel_bandit_data(
      num_datapoints,
      num_actions,
      context_dim,
      delta,
      mean_v,
      std_v,
      mu_large,
      std_large)
  return context_action_pairs, rewards


def procure_dataset(hparams, num_wheels, seed=0):
  """Samples the full dataset for pretraining GNPs."""
  np.random.seed(seed)

  all_context_action_pairs, all_rewards = [], []
  for _ in range(num_wheels):
    delta = np.random.uniform()
    context_action_pairs, rewards = get_single_wheel_data(
        hparams.num_target + hparams.num_context,
        hparams.num_actions,
        hparams.context_dim,
        delta)
    all_context_action_pairs.append(context_action_pairs)
    all_rewards.append(rewards)

  all_context_action_pairs = np.stack(all_context_action_pairs)
  all_rewards = np.stack(all_rewards)
  return all_context_action_pairs, all_rewards


def get_splits(dataset, n_context, batch_size, points_perm=True):
  """Splits the dataset into target and context sets."""
  full_x, full_y = dataset
  dataset_perm = np.random.permutation(len(full_x))[:batch_size]
  if points_perm:
    datapoints_perm = np.random.permutation(full_x.shape[1])
  else:
    datapoints_perm = np.arange(full_x.shape[1])

  target_x = tf.to_float(full_x[dataset_perm[:, None], datapoints_perm])
  target_y = tf.to_float(full_y[dataset_perm[:, None], datapoints_perm])
  context_x = target_x[:, :n_context, :]
  context_y = target_y[:, :n_context, :]
  unseen_targets = target_y[:, n_context:]
  return context_x, context_y, target_x, target_y, unseen_targets


def training_loop(train_dataset,
                  valid_dataset,
                  model,
                  hparams):
  """Trains an SNP for a fixed number of iterations."""
  optimizer_config = {'optimizer': hparams.optimizer(hparams.lr),
                      'max_grad_norm': hparams.max_grad_norm}
  num_context = hparams.num_context
  best_mse = np.inf
  step = tf.function(utils.mse_step.python_function)  # pytype: disable=module-attr

  for it in range(hparams.num_iterations):
    batch_train_data = get_splits(
        train_dataset,
        num_context,
        hparams.batch_size,
        points_perm=True)
    nll, mse, local_z_kl, global_z_kl = step(
        model,
        batch_train_data,
        optimizer_config)

    if it % hparams.print_every == 0:
      (batch_context_x,
       batch_context_y,
       batch_target_x,
       batch_target_y,
       batch_unseen_targets) = get_splits(valid_dataset,
                                          num_context,
                                          hparams.batch_size,
                                          points_perm=False)
      prediction = model(batch_context_x,
                         batch_context_y,
                         batch_target_x,
                         batch_target_y)

      batch_unseen_predictions = prediction[:, num_context:]
      valid_nll = utils.nll(batch_unseen_targets, batch_unseen_predictions)
      valid_mse = utils.mse(batch_unseen_targets, batch_unseen_predictions)
      if model.local_variational:
        valid_local_kl = tf.reduce_mean(
            tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
      else:
        valid_local_kl = 0.
      valid_global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))

      print('it: {}, train nll: {}, mse: {}, local kl: {} global kl: {} '
            'valid nll: {}, mse: {}, local kl: {} global kl: {}'
            .format(it, nll, mse, local_z_kl, global_z_kl,
                    valid_nll, valid_mse, valid_local_kl, valid_global_kl))
      if valid_mse.numpy() < best_mse:
        best_mse = valid_mse.numpy()
        print('Saving best model with MSE', best_mse)
        model.save_weights(hparams.save_path)


def train(data_hparams,
          model_hparams,
          training_hparams):
  """Executes the training pipeline for SNPs."""
  all_context_action_pairs, all_rewards = procure_dataset(
      data_hparams,
      num_wheels=100,
      seed=0)
  train_dataset = (all_context_action_pairs, all_rewards)

  all_context_action_pairs, all_rewards = procure_dataset(
      data_hparams,
      num_wheels=10,
      seed=42)
  valid_dataset = (all_context_action_pairs, all_rewards)

  model = regressor.Regressor(
      input_dim=data_hparams.context_dim + data_hparams.num_actions,
      output_dim=1,
      x_encoder_sizes=model_hparams.x_encoder_sizes,
      x_y_encoder_sizes=model_hparams.x_y_encoder_sizes,
      global_latent_net_sizes=model_hparams.global_latent_net_sizes,
      local_latent_net_sizes=model_hparams.local_latent_net_sizes,
      heteroskedastic_net_sizes=model_hparams.heteroskedastic_net_sizes,
      att_type=model_hparams.att_type,
      att_heads=model_hparams.att_heads,
      uncertainty_type=model_hparams.uncertainty_type,
      mean_att_type=model_hparams.mean_att_type,
      scale_att_type_1=model_hparams.scale_att_type_1,
      scale_att_type_2=model_hparams.scale_att_type_2,
      activation=model_hparams.activation,
      output_activation=model_hparams.output_activation,
      data_uncertainty=model_hparams.data_uncertainty,
      local_variational=model_hparams.local_variational)

  training_loop(train_dataset,
                valid_dataset,
                model,
                training_hparams)

