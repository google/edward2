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

"""Implements the attention layers.
"""

import tensorflow.compat.v1 as tf


def uniform_attention(q, v):
  """Uniform attention. Equivalent to np.

  Args:
    q: queries. Tensor of shape [B, m, d_k].
    v: values. Tensor of shape [B, n, d_v].

  Returns:
    tensor of shape [B, m, d_v].
  """
  total_points = tf.shape(q)[1]
  rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [B, 1, d_v]
  rep = tf.tile(rep, [1, total_points, 1])
  return rep


def laplace_attention(q, k,
                      v=None,
                      scale=1.,
                      normalize=True,
                      weights_only=False,
                      hard=False):
  """Computes laplace exponential attention.

  Args:
    q: queries. Tensor of shape [B, m, d_k].
    k: keys. Tensor of shape [B, n, d_k].
    v: values. Tensor of shape [B, n, d_v]. Can be None if weights_only=True.
    scale: Attn hyperparam that scales the L1 distance. Tensor of shape [B].
    normalize: Boolean that determines whether weights sum to 1.
    weights_only: Boolean which returns attention weights if True.
    hard: Returns one-hot argmax weights instead of softmax if True.

  Returns:
    tensor of shape [B, m, d_v].
  """
  k = tf.expand_dims(k, axis=1)  # [B, 1, n, d_k]
  q = tf.expand_dims(q, axis=2)  # [B, m, 1, d_k]
  d_k = tf.shape(q)[-1]
  scale = tf.reshape(scale * tf.sqrt(tf.cast(d_k, tf.float32)), [-1, 1, 1, 1])
  unnorm_weights = - tf.abs((k - q) / scale)  # [B, m, n, d_k]
  unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [B, m, n]
  if normalize:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: 1 + tf.tanh(x)

  weights = weight_fn(unnorm_weights)  # [B, m, n]
  if hard:
    weights = tf.one_hot(
        tf.math.argmax(weights, axis=-1),
        depth=tf.shape(k)[2],
        axis=-1)
  if weights_only:
    return weights
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B, m, d_v]
  return rep


def squared_exponential_attention(q, k,
                                  v=None,
                                  scale=1.,
                                  normalize=True,
                                  weights_only=False,
                                  hard=False):
  """Computes squared exponential attention.

  Args:
    q: queries. Tensor of shape [B, m, d_k].
    k: keys. Tensor of shape [B, n, d_k].
    v: values. Tensor of shape [B, n, d_v]. Can be None if weights_only=True.
    scale: Attn hyperparam that scales the L1 distance. Tensor of shape [B].
    normalize: Boolean that determines whether weights sum to 1.
    weights_only: Boolean which returns attention weights if True.
    hard: Returns one-hot argmax weights instead of softmax if True.

  Returns:
    tensor of shape [B, m, d_v].
  """
  k = tf.expand_dims(k, axis=1)  # [B, 1, n, d_k]
  q = tf.expand_dims(q, axis=2)  # [B, m, 1, d_k]
  scale = tf.reshape(scale, [-1, 1, 1])  # [B, 1, 1]
  unnorm_weights = tf.exp(-0.5 * tf.reduce_sum(tf.square(k - q), axis=-1)
                          / tf.square(scale))  # [B, m, n, d_k]
  if normalize:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: x
  weights = weight_fn(unnorm_weights)  # [B, m, n]
  if hard:
    weights = tf.one_hot(
        tf.math.argmax(weights, axis=-1),
        depth=tf.shape(k)[2],
        axis=-1)
  if weights_only:
    return weights
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B, m, d_v]
  return rep


def dot_product_attention(q, k,
                          v=None,
                          scale=1.,
                          normalize=True,
                          weights_only=False,
                          hard=False):
  """Computes dot product attention.

  Args:
    q: queries. Tensor of shape [B, m, d_k].
    k: keys. Tensor of shape [B, n, d_k].
    v: values. Tensor of shape [B, n, d_v]. Can be None if weights_only=True.
    scale: Attn hyperparam that scales the dot product. Tensor of shape [B].
    normalize: Boolean that determines whether weights sum to 1.
    weights_only: Boolean which returns attention weights if True.
    hard: Returns one-hot argmax weights instead of softmax if True.

  Returns:
    tensor of shape [B, m, d_v].
  """
  d_k = tf.shape(q)[-1]
  scale = tf.reshape(scale * tf.sqrt(tf.cast(d_k, tf.float32)),
                     [-1, 1, 1])  # [B, 1, 1]
  unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [B, m, n]
  if normalize:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = tf.sigmoid
  weights = weight_fn(unnorm_weights)  # [B, m, n]
  if hard:
    weights = tf.one_hot(
        tf.math.argmax(weights, axis=-1),
        depth=tf.shape(k)[1],
        axis=-1)
  if weights_only:
    return weights
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [B, m, d_v]
  return rep


def multihead_attention(projection_nets,
                        q,
                        k,
                        v,
                        num_heads=8,
                        scale=1.,
                        normalize=True):
  """Computes multi-head attention.

  Args:
    projection_nets: List of lists of projection_nets for q, k, v, dot_product
      output for each head.
    q: queries. Tensor of  shape [B, m, d_k].
    k: keys. Tensor of shape [B, n, d_k].
    v: values. Tensor of shape [B, n, d_v].
    num_heads: number of heads. Should divide d_v.
    scale: Attn hyperparam that scales the dot product. Tensor of shape [B].
    normalize: Boolean that determines whether weights sum to 1.

  Returns:
    tensor of shape [B, m, d_v].
  """
  rep = tf.constant(0.0)
  for h in range(num_heads):
    q_net, k_net, v_net, r_net = projection_nets[h]
    o = dot_product_attention(
        q_net(q),
        k_net(k),
        v_net(v),
        scale=scale,
        normalize=normalize)
    rep += r_net(o)
  return rep


class AttentionLayer(tf.python.keras.layers.Layer):
  """The Attention module."""

  def __init__(self,
               att_type,
               scale=1.,
               normalize=True,
               num_heads=8):
    """Create attention module.

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.
    Args:
      att_type: type of attention. One of the following:
          ['uniform','laplace','dot_product','multihead']
      scale: scale of attention.
      normalize: Boolean determining whether to:
          1. apply softmax to weights so that they sum to 1 across context pts
          2. apply custom transformation to have weights in [0,1].
      num_heads: number of heads for multihead.
    """
    super(AttentionLayer, self).__init__()
    self._type = att_type
    self._scale = scale
    self._normalize = normalize
    if self._type == 'multihead':
      self._num_heads = num_heads

  def build(self, input_shape):
    assert isinstance(input_shape, list)
    d_k, d_v = input_shape

    if self._type == 'multihead':
      num_heads = self._num_heads
      head_size = int(d_v / num_heads)
      key_initializer = tf.random_normal_initializer(stddev=d_k**-0.5)
      value_initializer = tf.random_normal_initializer(stddev=d_v**-0.5)
      self.multihead_nets = []

      for h in range(num_heads):
        query_net = tf.python.keras.Sequential(
            [tf.python.keras.layers.InputLayer([None, d_k]),
             tf.python.keras.layers.Conv1D(head_size, 1,
                                    kernel_initializer=key_initializer,
                                    name='wq%d' % h, use_bias=False,
                                    padding='VALID')])
        key_net = tf.python.keras.Sequential(
            [tf.python.keras.layers.InputLayer([None, d_k]),
             tf.python.keras.layers.Conv1D(head_size, 1,
                                    kernel_initializer=key_initializer,
                                    name='wk%d' % h, use_bias=False,
                                    padding='VALID')])
        value_net = tf.python.keras.Sequential(
            [tf.python.keras.layers.InputLayer([None, d_v]),
             tf.python.keras.layers.Conv1D(head_size, 1,
                                    kernel_initializer=key_initializer,
                                    name='wv%d' % h, use_bias=False,
                                    padding='VALID')])
        rep_net = tf.python.keras.Sequential(
            [tf.python.keras.layers.InputLayer([None, head_size]),
             tf.python.keras.layers.Conv1D(d_v, 1,
                                    kernel_initializer=value_initializer,
                                    name='wo%d' % h, use_bias=False,
                                    padding='VALID')])
        self.multihead_nets.append([query_net, key_net, value_net, rep_net])

    super(AttentionLayer, self).build(input_shape)

  def __call__(self, q, k, v):
    """Apply attention to create aggregated representation of v.

    Args:
      q: Tensor of shape [B, m, d_x].
      k: Tensor of shape [B, n, d_x].
      v: Tensor of shape [B, n, d].

    Returns:
      Tensor of shape [B, m, d].

    Raises:
      NameError: The argument for type was invalid.
    """

    if self._type == 'uniform':
      rep = uniform_attention(q, v)
    elif self._type == 'laplace':
      rep = laplace_attention(q, k, v, self._scale, self._normalize)
    elif self._type == 'dot_product':
      rep = dot_product_attention(q, k, v, self._scale, self._normalize)
    elif self._type == 'multihead':
      rep = multihead_attention(
          self.multihead_nets,
          q,
          k,
          v,
          self._num_heads,
          self._scale,
          self._normalize)
    else:
      raise NameError(("'att_type' not among ['uniform','laplace','dot_product'"
                       ",'multihead']"))

    return rep
