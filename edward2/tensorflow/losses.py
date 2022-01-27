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

"""Specialized loss functions to be used with Edward2 models.

This module contains the uncertainty-aware cross-entropy loss, as used by
Biloš et al. [1] and Charpentier et al. [2]. It is used in conjuction with
the PosteriorNetworkLayer in ed.layers.

[1] Biloš, M., Charpentier, B., & Günnemann, S. (2019).
Uncertainty on asynchronous time event prediction.
arXiv preprint arXiv:1911.05503.
[2] Charpentier, B., Zügner, D., & Günnemann, S. (2020).
Posterior network: Uncertainty estimation without OOD samples via
density-based pseudo-counts. arXiv preprint arXiv:2006.09239.
"""

import functools

import tensorflow as tf


def uce_loss(entropy_reg=1e-5, sparse=True,
             return_all_loss_terms=False, num_classes=None):
  """Computes the UCE loss, either from sparse or dense labels."""
  if sparse and num_classes is None:
    raise ValueError('Number of classes must be defined for the sparse loss.')
  if sparse:
    return functools.partial(_sparse_uce_loss,
                             entropy_reg=entropy_reg,
                             return_all_loss_terms=return_all_loss_terms,
                             num_classes=num_classes)
  else:
    return functools.partial(_uce_loss,
                             entropy_reg=entropy_reg,
                             return_all_loss_terms=return_all_loss_terms)


def _uce_loss(labels, alpha, entropy_reg=1e-5, return_all_loss_terms=False):
  """UCE loss, as in the Posterior Network paper.

  Args:
    labels: Numpy array or Tensor of true labels for the data points.
    alpha: Predicted Dirichlet parameters for the data points.
    entropy_reg: Entropy regularizer to weigh the entropy term in the loss.
    return_all_loss_terms: Flag to return the separate loss terms.

  Returns:
    Either the scalar total loss or a tuple of the different loss terms.
  """
  # This computes the normalizer of the Dirichlet distribution.
  alpha_0 = tf.reduce_sum(alpha, axis=-1, keepdims=True) * tf.ones_like(alpha)
  # This computes the Dichlet entropy.
  entropy = _dirichlet_entropy(alpha)
  # This computes the expected cross-entropy under the Dirichlet.
  # (Eq. 10 in paper [2])
  ce_loss = tf.reduce_sum(
      (tf.math.digamma(alpha_0) - tf.math.digamma(alpha)) * labels, axis=-1)
  # The UCE loss is E_q[CE(p,y)] - lambda * H(q).
  loss = ce_loss - entropy_reg * entropy
  if return_all_loss_terms:
    return tf.reduce_mean(loss), tf.reduce_mean(ce_loss), tf.reduce_mean(
        entropy)
  else:
    return tf.reduce_mean(loss)


def _sparse_uce_loss(labels,
                     alpha,
                     num_classes,
                     entropy_reg=1e-5,
                     return_all_loss_terms=False):
  """UCE loss with sparse labels. Same args as _uce_loss."""
  labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
  return _uce_loss(labels_one_hot, alpha, entropy_reg, return_all_loss_terms)


def _dirichlet_entropy(alpha):
  """Computes the entropy of a Dirichlet distribution (Eq. 11 in paper [2])."""
  k = tf.cast(tf.shape(alpha)[-1], alpha.dtype)
  total_concentration = tf.reduce_sum(alpha, axis=-1)
  entropy = (
      tf.math.lbeta(alpha) +
      ((total_concentration - k) * tf.math.digamma(total_concentration)) -
      tf.reduce_sum((alpha - 1.) * tf.math.digamma(alpha), axis=-1))
  return entropy
