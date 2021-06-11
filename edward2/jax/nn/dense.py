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
    use_ensemble_bias: whether to add a bias to the BE output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  ens_size: int
  activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
  use_ensemble_bias: bool = True
  dtype: Optional[DType] = None
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = nn.initializers.xavier_uniform()
  bias_init: InitializeFn = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    """Applies layer to input.

    Args:
      inputs: jnp.ndarray of shape [ens_size * batch_size, ..., input_dim].

    Returns:
      jnp.ndarray of shape [ens_size * batch_size, ..., features].
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

    inputs_shape = inputs.shape
    inputs = jnp.reshape(inputs, (self.ens_size, -1) + inputs_shape[1:])
    outputs = jnp.einsum('E...C,EC,CD,ED->E...D', inputs, alpha, kernel, gamma)

    if self.use_ensemble_bias:
      bias = self.param('bias', self.bias_init, (self.ens_size, self.features),
                        dtype)
      bias_shape = (self.ens_size,) + (1,) * (outputs.ndim - 2) + (
          self.features,)
      outputs = outputs + jnp.reshape(bias, bias_shape)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return jnp.reshape(outputs, inputs_shape[:-1] + (self.features,))
