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

"""Defined utility functions for attentive uncertainty models.
"""

import tensorflow.compat.v1 as tf


def mse(y_true, y_pred_dist, reduction='mean'):
  """Returns the mean squared error for a predictive distribution.

  Args:
    y_true: (float) Tensor of target labels.
    y_pred_dist: An edward2 distribution object.
    reduction: (string) Either 'sum' or 'mean'.
  """
  if reduction == 'sum':
    return tf.reduce_sum(
        tf.squared_difference(y_true,
                              y_pred_dist.distribution.mean()))
  else:
    return tf.losses.mean_squared_error(y_true,
                                        y_pred_dist.distribution.mean())


def nll(y_true, y_pred_dist, reduction='mean'):
  """Returns the negative log-likelihood of a model w.r.t. true targets.

  Args:
    y_true: (float) Tensor of target labels.
    y_pred_dist: An edward2 distribution object.
    reduction: (string) Either 'sum' or 'mean'.
  """
  log_p = y_pred_dist.distribution.log_prob(y_true)
  if reduction == 'sum':
    return -tf.reduce_sum(log_p)
  else:
    return -tf.reduce_mean(log_p)


def mlp_block(in_dim, hidden_sizes, activation=tf.nn.relu):
  """Return keras sequential MLP object for the final axis of a 2/3D tensor.

  Args:
    in_dim: (int) Input dimension for final axis.
    hidden_sizes: (list of ints) An iterable containing the output sizes of the
      MLP as defined in `basic.Linear`.
    activation: (callable) Activation applied to all but the final layer.

  Returns:
    tensor of shape [B, n, d_out] where d_out = hidden_sizes[-1]
  """

  net = tf.python.keras.Sequential([tf.python.keras.layers.InputLayer(in_dim)])
  for size in hidden_sizes[:-1]:
    net.add(tf.python.keras.layers.Dense(size, activation=activation))
  net.add(tf.python.keras.layers.Dense(hidden_sizes[-1], activation=None))
  return net


@tf.function
def train_step(model, data, optimizer_config, is_mse=False):
  """Applies gradient updates and returns appropriate metrics.

  Args:
    model: An instance of SNP Regressor.
    data: A 5-tuple consisting of context_x, context_y, target_x, target_y,
      unseen_targets (i.e., target_x-context_x).
    optimizer_config: A dictionary with two keys: an 'optimizer' object and
      a 'max_grad_norm' for clipping gradients.
    is_mse: Use mse (fixed variance) if True else use nll.

  Returns:
    nll_term: Negative log-likelihood assigned by model to unseen targets.
    mse_term: Mean squared error of model for unseen targets.
    local_kl: KL loss for latent variables of unseen targets.
    global_kl: KL loss for global latent variable.
  """
  context_x, context_y, target_x, target_y, unseen_targets = data
  num_context = tf.shape(context_x)[1]
  with tf.GradientTape() as tape:
    prediction = model(
        context_x,
        context_y,
        target_x,
        target_y)
    unseen_predictions = prediction[:, num_context:]
    nll_term = nll(unseen_targets, unseen_predictions)
    mse_term = mse(unseen_targets, unseen_predictions)
    loss = mse_term if is_mse else nll_term
    if model.local_variational:
      local_kl = tf.reduce_mean(
          tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
    else:
      local_kl = 0.
    global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))
    loss += local_kl + global_kl
  gradients = tape.gradient(loss, model.trainable_variables)
  max_grad_norm = optimizer_config['max_grad_norm']
  optimizer = optimizer_config['optimizer']
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
  return nll_term, mse_term, local_kl, global_kl


@tf.function
def train_gnp_step(model, data, optimizer_config, is_mse=False):
  """Applies gradient updates and returns appropriate metrics.

  Args:
    model: An instance of GNP Regressor.
    data: A 5-tuple consisting of context_x, context_y, target_x, target_y,
      unseen_targets (i.e., target_x-context_x).
    optimizer_config: A dictionary with two keys: an 'optimizer' object and
      a 'max_grad_norm' for clipping gradients.
    is_mse: Use mse (fixed variance) if True else use nll.

  Returns:
    nll_term: Negative log-likelihood assigned by model to unseen targets.
    mse_term: Mean squared error of model for unseen targets.
    local_kl: KL loss for latent variables of unseen targets.
    global_kl: KL loss for global latent variable.
  """
  context_x, context_y, target_x, target_y, unseen_targets = data
  num_context = tf.shape(context_x)[1]
  with tf.GradientTape() as tape:
    prediction = model(
        context_x,
        context_y,
        target_x,
        target_y)
    unseen_predictions = prediction[:, num_context:]
    nll_term = nll(unseen_targets, unseen_predictions)
    mse_term = mse(unseen_targets, unseen_predictions)
    loss = mse_term if is_mse else nll_term
    local_kl = tf.reduce_mean(
        tf.reduce_sum(model.losses[-1][:, num_context:], axis=[1, 2]))
    global_kl = tf.reduce_mean(tf.reduce_sum(model.losses[-2], axis=-1))
    loss += local_kl + global_kl
  gradients = tape.gradient(loss, model.trainable_variables)
  max_grad_norm = optimizer_config['max_grad_norm']
  optimizer = optimizer_config['optimizer']
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
  optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
  return nll_term, mse_term, local_kl, global_kl

