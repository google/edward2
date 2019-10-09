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

"""Defined utility functions for contextual bandits via SNPs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from deep_contextual_bandits import contextual_bandit  # local file import


def get_data_with_masked_rewards(dataset):
  """Returns all observations with one-hot weights for actions."""
  weights = np.zeros((dataset.contexts.shape[0], dataset.num_actions))
  a_ind = np.array(list(enumerate(dataset.actions)))
  weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
  masked_rewards = dataset.rewards[np.arange(len(dataset.actions)),
                                   dataset.actions]
  return dataset.contexts, masked_rewards, weights


def get_batch_with_masked_rewards(dataset, batch_size, with_replacement=True):
  """Returns a random mini-batch with one-hot weights for actions."""
  n, _ = dataset.contexts.shape
  if dataset.buffer_s == -1:
    # Use all the data.
    ind = np.random.choice(range(n), batch_size, replace=with_replacement)
  else:
    # Use only buffer (last buffer_s obs).
    ind = np.random.choice(range(max(0, n - dataset.buffer_s), n), batch_size,
                           replace=with_replacement)

  weights = np.zeros((batch_size, dataset.num_actions))
  sampled_actions = np.array(dataset.actions)[ind]
  a_ind = np.array(list(enumerate(sampled_actions)))
  weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
  masked_rewards = dataset.rewards[ind, sampled_actions]
  return dataset.contexts[ind, :], masked_rewards, weights


def run_contextual_bandit(context_dim,
                          num_actions,
                          dataset,
                          algos,
                          num_contexts=None):
  """Run a contextual bandit problem on a set of algorithms.

  Args:
    context_dim: Dimension of the context.
    num_actions: Number of available actions.
    dataset: Matrix where every row is a context + num_actions rewards.
    algos: List of algorithms to use in the contextual bandit instance.
    num_contexts: Number of contexts.

  Returns:
    h_actions: Matrix with actions: size (num_context, num_algorithms).
    h_rewards: Matrix with rewards: size (num_context, num_algorithms).
  """
  if num_contexts is None:
    num_contexts = dataset.shape[0]

  # Create contextual bandit
  cmab = contextual_bandit.ContextualBandit(context_dim, num_actions)
  cmab.feed_data(dataset)

  h_actions = np.empty((0, len(algos)), float)
  h_rewards = np.empty((0, len(algos)), float)

  # Run the contextual bandit process
  for i in range(num_contexts):
    context = cmab.context(i)
    actions = [a.action(context) for a in algos]
    rewards = [cmab.reward(i, action) for action in actions]

    for j, a in enumerate(algos):
      a.update(context, actions[j], rewards[j])

    h_actions = np.vstack((h_actions, np.array(actions)))
    h_rewards = np.vstack((h_rewards, np.array(rewards)))

  return h_actions, h_rewards


def display_results(algo_names, per_timestep_regrets, time, name):
  """Displays summary statistics of the performance of each algorithm."""
  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(name, time))
  print('---------------------------------------------------')

  performance_pairs = []
  for j, a in enumerate(algo_names):
    performance_pairs.append((a, np.sum(per_timestep_regrets[:, j])))
  performance_pairs = sorted(performance_pairs, key=lambda elt: elt[1])
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}){:20}|\t\t cummulative regret = {:10}.'.format(i, name, reward))
  print('---------------------------------------------------')
  print('---------------------------------------------------')

  sim_performance_pairs = []
  for j, a in enumerate(algo_names):
    sim_performance_pairs.append((a, np.sum(per_timestep_regrets[-500:, j])))
  sim_performance_pairs = sorted(sim_performance_pairs, key=lambda elt: elt[1])
  for i, (name, reward) in enumerate(sim_performance_pairs):
    print('{:3}) {:20}|\t\t simple regret = {:10}.'.format(i, name, reward))
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def mse(y_true, y_pred_dist, actions):
  """Returns the mean squared error for a predictive distribution.

  Args:
    y_true: (float) Tensor of target labels.
    y_pred_dist: A tfp distribution object.
    actions: One-hot masking actions.
  """
  return tf.losses.mean_squared_error(y_true,
                                      actions*y_pred_dist.distribution.mean())


def nll(y_true, y_pred_dist, actions):
  """Returns the negative log-likelihood of a model w.r.t. true targets.

  Args:
    y_true: (float) Tensor of target labels.
    y_pred_dist: An edward2 distribution object.
    actions: One-hot masking actions.
  """
  log_p = actions * y_pred_dist.distribution.log_prob(y_true)
  return -tf.reduce_mean(tf.reduce_sum(log_p, axis=-1))


@tf.function
def mse_anp_step(model, data, optimizer_config):
  """Applies gradient updates and returns appropriate metrics.

  Args:
    model: An instance of SNP Regressor.
    data: A 5-tuple consisting of context_x, context_y, target_x, target_y,
      unseen_targets (i.e., target_x-context_x).
    optimizer_config: A dictionary with two keys: an 'optimizer' object and
      a 'max_grad_norm' for clipping gradients.

  Returns:
    mse_term: Mean squared error of model for unseen targets.
    local_kl: KL loss for latent variables of unseen targets.
    global_kl: KL loss for global latent variable.
  """
  (context_x,
   context_y,
   target_x,
   target_y,
   unseen_target_y,
   unseen_target_a) = data
  num_context = tf.shape(context_x)[1]
  with tf.GradientTape() as tape:
    prediction = model(context_x,
                       context_y,
                       target_x,
                       target_y)
    unseen_predictions = prediction[:, num_context:]
    mse_term = mse(unseen_target_y, unseen_predictions, unseen_target_a)
    if model.local_variational:
      local_kl = tf.reduce_mean(
          tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
    else:
      local_kl = 0.
    global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))
    loss = mse_term + local_kl + global_kl
  gradients = tape.gradient(loss, model.trainable_variables)
  max_grad_norm = optimizer_config['max_grad_norm']
  optimizer = optimizer_config['optimizer']
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
  return mse_term, local_kl, global_kl


@tf.function
def mse_gnp_step_bandits(model, data, optimizer_config):
  """Applies gradient updates and returns appropriate metrics.

  Args:
    model: An instance of SNP Regressor.
    data: A 5-tuple consisting of context_x, context_y, target_x, target_y,
      unseen_targets (i.e., target_x-context_x).
    optimizer_config: A dictionary with two keys: an 'optimizer' object and
      a 'max_grad_norm' for clipping gradients.

  Returns:
    mse_term: Mean squared error of model for unseen targets.
    local_kl: KL loss for latent variables of unseen targets.
    global_kl: KL loss for global latent variable.
  """
  (context_x,
   context_y,
   target_x,
   target_y,
   unseen_target_y,
   unseen_target_a) = data
  num_context = tf.shape(context_x)[1]
  with tf.GradientTape() as tape:
    prediction = model(context_x,
                       context_y,
                       target_x,
                       target_y)
    unseen_predictions = prediction[:, num_context:]
    mse_term = mse(unseen_target_y, unseen_predictions, unseen_target_a)
    local_kl = tf.reduce_mean(
        tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
    global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))
    loss = mse_term + local_kl + global_kl
  gradients = tape.gradient(loss, model.trainable_variables)
  max_grad_norm = optimizer_config['max_grad_norm']
  optimizer = optimizer_config['optimizer']
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
  return mse_term, local_kl, global_kl


@tf.function
def nll_gnp_step_bandits(model, data, optimizer_config):
  """Applies gradient updates and returns appropriate metrics.

  Args:
    model: An instance of SNP Regressor.
    data: A 5-tuple consisting of context_x, context_y, target_x, target_y,
      unseen_targets (i.e., target_x-context_x).
    optimizer_config: A dictionary with two keys: an 'optimizer' object and
      a 'max_grad_norm' for clipping gradients.

  Returns:
    nll_term: Negative log-likelihood of model for unseen targets.
    local_kl: KL loss for latent variables of unseen targets.
    global_kl: KL loss for global latent variable.
  """
  (context_x,
   context_y,
   target_x,
   target_y,
   unseen_target_y,
   unseen_target_a) = data
  num_context = tf.shape(context_x)[1]
  with tf.GradientTape() as tape:
    prediction = model(context_x,
                       context_y,
                       target_x,
                       target_y)
    unseen_predictions = prediction[:, num_context:]
    nll_term = nll(unseen_target_y, unseen_predictions, unseen_target_a)
    local_kl = tf.reduce_mean(
        tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
    global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))
    loss = nll_term + local_kl + global_kl
  gradients = tape.gradient(loss, model.trainable_variables)
  max_grad_norm = optimizer_config['max_grad_norm']
  optimizer = optimizer_config['optimizer']
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
  return nll_term, local_kl, global_kl
