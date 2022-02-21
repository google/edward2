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

"""Uncertainty-based dense layers in JAX."""

from typing import Iterable, Callable, Optional

import flax.linen as nn
import jax.numpy as jnp

DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]


class DenseBatchEnsemble(nn.Module):
  """A batch ensemble dense layer.

  Attributes:
    features: the number of output features.
    ens_size: the number of ensemble members.
    activation: activation function.
    use_bias: whether to add a bias to the BE output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  ens_size: int
  activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  use_bias: bool = True
  dtype: Optional[DType] = None
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = nn.initializers.lecun_normal()
  bias_init: InitializeFn = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs,
               is_first_be: bool = False,
               index: Optional[int] = None):
    """Applies layer to input.

    Args:
      inputs: jnp.ndarray of shape [ens_size, batch_size, ..., input_dim] or
      [batch_size, ..., input_dim] if is_first_be.
      is_first_be:
      index:

    Returns:
      jnp.ndarray of shape [ens_size, batch_size, ..., features].
    """
    dtype = self.dtype or inputs.dtype
    inputs = jnp.asarray(inputs, dtype)
    input_dim = inputs.shape[-1]

    kernel = self.param('kernel', self.kernel_init, (input_dim, self.features),
                        dtype)
    alpha = self.param('fast_weight_alpha', self.alpha_init,
                       (self.ens_size, input_dim), dtype)
    gamma = self.param('fast_weight_gamma', self.gamma_init,
                       (self.ens_size, self.features), dtype)

    if index is None:
      if not is_first_be:
        outputs = jnp.einsum(
            'E...C,EC,CD,ED->E...D', inputs, alpha, kernel, gamma)
      else:
        # TODO(trandustin): Testing einsum instead of tile.
        outputs = jnp.einsum('...C,EC,CD,ED->E...D', inputs, alpha, kernel, gamma)

      if self.use_bias:
        bias = self.param('bias', self.bias_init, (self.ens_size, self.features),
                          dtype)
        bias_shape = (self.ens_size,) + (1,) * (outputs.ndim - 2) + (
            self.features,)
        # TODO(trandustin): When finetuned from a deterministic upstream ckpt,
        # need to enable setting bias to use ens_size=1 version as below so it can
        # be set to deterministic's bias. Or ensemble of biases should at least be
        # initialized that way.
        # bias = self.param('bias', self.bias_init, (self.features,), dtype)
        # bias_shape = (1,) * (outputs.ndim - 1) + (self.features,)
        outputs = outputs + jnp.reshape(bias, bias_shape)
    else:
      alphai = alpha[index]
      gammai = gamma[index]
      outputs = jnp.einsum('...C,C,CD,D->...D', inputs, alphai, kernel, gammai)
      if self.use_bias:
        # For stochastic BE, we use a shared bias across members for now. This
        # makes bias training easier although less diversity that may be
        # important.
        bias = self.param('bias', self.bias_init, (self.features,), dtype)
        bias_shape = (1,) * (outputs.ndim - 1) + (self.features,)
        outputs = outputs + jnp.reshape(bias, bias_shape)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return outputs
