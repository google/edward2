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

"""Uncertainty-based attention layers in JAX."""

from typing import Any, Callable, Optional, Sequence

from edward2.jax.nn import dense
import flax.linen as nn
from flax.linen.linear import default_kernel_init
from jax import lax
import jax.numpy as jnp

PRNGKey = Any
Shape = Sequence[int]
Dtype = Any
Array = Any
InitializeFn = Callable[[PRNGKey, Shape, Dtype], Array]


class MultiHeadDotProductAttentionBE(nn.Module):
  """BatchEnsemble of multi-head dot-product attention layers.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      param_dtype: the dtype passed to parameter initializers (default:
        float32).
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
  """
  num_heads: int
  ens_size: int
  dtype: Dtype = jnp.float32
  # TODO(trandustin): param_dtype is currently unused.
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  out_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.
  deterministic: Optional[bool] = None
  # TODO(trandustin): precision is currently unused.
  precision: Any = None
  kernel_init: InitializeFn = default_kernel_init
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  bias_init: InitializeFn = nn.initializers.zeros
  use_bias: bool = True
  attention_fn: Callable[
      [Array, Array, Array], Array] = nn.dot_product_attention
  decode: bool = False

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               deterministic: Optional[bool] = None):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    Args:
      inputs_q: input queries of shape
        `[batch_sizes..., length, features]`.
      inputs_kv: key/values of shape
        `[batch_sizes..., length, features]`.
      mask: attention mask of shape
        `[batch_sizes..., num_heads, query_length, key/value_length]`.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.

    Returns:
      output of shape `[batch_sizes..., length, features]`.
    """
    if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
      deterministic = nn.merge_param('deterministic',
                                     self.deterministic,
                                     deterministic)
    features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads

    def dense_fn(name):
      dense_lyr = dense.DenseBatchEnsemble(
          features=self.num_heads * head_dim,
          ens_size=self.ens_size,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          name=name)
      def f(x, *args, **kwargs):
        x = dense_lyr(x, *args, **kwargs)
        x = jnp.reshape(x, x.shape[:-1] + (self.num_heads, head_dim))
        return x
      return f
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query, key, value = (dense_fn(name='query')(inputs_q),
                         dense_fn(name='key')(inputs_kv),
                         dense_fn(name='value')(inputs_kv))

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable('cache', 'cached_key',
                                 jnp.zeros, key.shape, key.dtype)
      cached_value = self.variable('cache', 'cached_value',
                                   jnp.zeros, value.shape, value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        *batch_dims, max_length, num_heads, depth_per_head = (
            cached_key.value.shape)
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = nn.combine_masks(
            mask,
            jnp.broadcast_to(jnp.arange(max_length) <= cur_index,
                             tuple(batch_dims) + (1, 1, max_length)))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # apply attention
    x = self.attention_fn(
        query,
        key,
        value,
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=self.precision)  # pytype: disable=wrong-keyword-args
    # back to the original inputs dimensions
    def dense_fn2(name):
      dense_lyr = dense.DenseBatchEnsemble(
          features=features,
          ens_size=self.ens_size,
          kernel_init=self.kernel_init,
          alpha_init=self.alpha_init,
          gamma_init=self.gamma_init,
          bias_init=self.bias_init,
          use_bias=self.use_bias,
          dtype=self.dtype,
          name=name)
      def f(x, *args, **kwargs):
        x = jnp.reshape(x, x.shape[:-2] + (x.shape[-2] * x.shape[-1],))
        x = dense_lyr(x, *args, **kwargs)
        return x
      return f
    out = dense_fn2(name='out')(x)
    return out
