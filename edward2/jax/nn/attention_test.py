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

"""Tests for flax.deprecated.nn.attention."""

from absl.testing import absltest
from absl.testing import parameterized

import edward2.jax as ed
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


class AttentionTest(parameterized.TestCase):

  def test_multihead_self_attention_single_ensemble(self):
    """Tests that Flax's MHA == MHA ensemble with ens_size=1."""
    rng = jax.random.PRNGKey(0)
    rng, next_rng = jax.random.split(rng)
    x = jax.random.normal(rng, (4, 2, 3, 5))
    attention = nn.MultiHeadDotProductAttention(
        num_heads=8,
        qkv_features=16,
        deterministic=False,
    )
    attention_batch_ensemble = ed.nn.MultiHeadDotProductAttentionBE(
        num_heads=8,
        ens_size=1,
        qkv_features=16,
        deterministic=False,
        alpha_init=jax.nn.initializers.ones,
        gamma_init=jax.nn.initializers.ones,
    )
    rng, _ = jax.random.split(next_rng)
    y1, _ = attention.init_with_output(rng, x, x)
    y2, _ = attention_batch_ensemble.init_with_output(rng, x, x)
    np.testing.assert_array_equal(y1, y2)

  def test_multihead_encoder_decoder_attention(self):
    rng = jax.random.PRNGKey(0)
    q = jnp.ones((4, 2, 3, 5))
    kv = jnp.ones((4, 2, 3, 5))
    sa_module = ed.nn.MultiHeadDotProductAttentionBE(
        num_heads=8,
        ens_size=2,
        qkv_features=16,
        kernel_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros,
        deterministic=False,
    )
    y, _ = sa_module.init_with_output(rng, q, kv)
    self.assertEqual(y.shape, q.shape)

  def test_multihead_self_attention_w_dropout(self):
    rng = jax.random.PRNGKey(0)
    x = jnp.ones((4, 2, 3, 5))
    sa_module = ed.nn.MultiHeadDotProductAttentionBE(
        num_heads=8,
        ens_size=2,
        qkv_features=16,
        kernel_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros,
        dropout_rate=0.1,
        deterministic=False,
    )
    rng1, rng2 = jax.random.split(rng)
    rngs = {'params': rng1, 'dropout': rng2}
    y, _ = sa_module.init_with_output(rngs, x, x)
    self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
  absltest.main()
