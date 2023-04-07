# coding=utf-8
# Copyright 2023 The Edward2 Authors.
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

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import flax.linen as nn
from jax import lax
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
InitializeFn = Callable[[PRNGKey, Shape, Dtype], Array]

default_kernel_init = nn.initializers.lecun_normal()


# taken from flax/linen/linear.py
def _normalize_axes(axes: Tuple[int, ...], ndim: int) -> Tuple[int, ...]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple(sorted(ax if ax >= 0 else ndim + ax for ax in axes))


# taken from flax/linen/linear.py
def _canonicalize_tuple(x: Union[Sequence[int], int]) -> Tuple[int, ...]:
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneralBatchEnsemble(nn.Module):
  """A BatchEnsemble linear transformation with flexible axes.

  Suppose the input has shape `batch_shape + (ens_size, num_examples) +
  axis_shape`. For example, `batch_shape` might be empty (unless you want a
  batch of dense layers) and `axis_shape` might be a 1-D input dimension. The
  layer's `batch_dims` corresponds to the axes of `batch_shape` and `axis`
  corresponds to the axes of `axis_shape`.

  The layer's parameters have the following shapes:

  * `kernel`: `batch_shape + axis_shape + features`
  * `fast_weight_alpha`: `batch_shape + (ens_size,) + axis_shape`
  * `fast_weight_gamma`: `batch_shape + (ens_size,) + features`
  * `bias`: `batch_shape + (ens_size,) + features`

  `DenseGeneral` is a wrapper around `jax.lax.dot_general`.

  Attributes:
    features: int or tuple with number of output features.
    ens_size: the number of ensemble members.
    axis: int or tuple with axes to apply the transformation on. For instance,
      (-2, -1) will apply the transformation to the last two axes.
    batch_dims: tuple with batch axes. It refers to batch as in "batched
      matmul", where a batch of inputs is applied over a corresponding batch of
      kernels. It is not the same as a typical batch size in machine learning
      where the kernel is shared across examples.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
  """
  features: Union[int, Sequence[int]]
  ens_size: int
  axis: Union[int, Sequence[int]] = -1
  batch_dims: Sequence[int] = ()
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = default_kernel_init
  bias_init: InitializeFn = nn.initializers.zeros
  precision: PrecisionLike = None

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

     The input's ens_size dimension can be 1. In this case, ensemble member
     calculations will be broadcasted. For example, an input of shape
     (1, batch_size, axis) will have output (ens_size, batch_size, features).
     This is useful when a BatchEnsemble layer is first applied in a model as
     there is no need for an explicit tiling operation.

    Args:
      inputs: The nd-array to be transformed. It has shape `batch_shape +
          (ens_size,...)`, where `[...]` includes dimensions that will be
          changed to `features`. For example, the layer might use an empty
          `batch_dims` and `axis=-1`, and the input has shape `(ens_size,
          num_examples, length, input_dim)`.

    Returns:
      The transformed input, of shape `batch_shape + (ens_size, ...)`, where
      `[...]` includes dimensions that have changed to `features`. In the above
      example, the output has shape `(ens_size, num_examples, length) +
      features`.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)
    batch_dims = _canonicalize_tuple(self.batch_dims)
    if batch_dims:
      max_dim = np.max(batch_dims)
      if set(batch_dims) != set(range(max_dim + 1)):
        raise ValueError('batch_dims %s must be consecutive leading '
                         'dimensions starting from 0.' % str(batch_dims))

    ndim = inputs.ndim
    n_batch_dims = len(batch_dims)
    n_batch_ens_dims = n_batch_dims + 1
    axis = _normalize_axes(axis, ndim)
    batch_dims = _normalize_axes(batch_dims, ndim)
    n_axis, n_features = len(axis), len(features)

    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
      flat_shape = (np.prod(shape[n_batch_dims:n_axis + n_batch_dims]),
                    np.prod(shape[-n_features:]),)
      kernel = jnp.concatenate([self.kernel_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
      return jnp.reshape(kernel, shape)
    def alpha_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
      flat_shape = (self.ens_size,
                    np.prod(shape[n_batch_ens_dims:n_axis + n_batch_ens_dims]),)
      kernel = jnp.concatenate([self.alpha_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
      return jnp.reshape(kernel, shape)
    def gamma_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
      flat_shape = (self.ens_size,
                    np.prod(shape[-n_features:]),)
      kernel = jnp.concatenate([self.gamma_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
      return jnp.reshape(kernel, shape)

    batch_shape = tuple(inputs.shape[ax] for ax in batch_dims)
    # batch and non-contracting dims of input with 1s for batch dims.
    expanded_batch_shape = batch_shape + (self.ens_size,)
    remainder = inputs.ndim - (n_batch_dims + 1 + n_axis)
    if remainder > 0:
      expanded_batch_shape = expanded_batch_shape + (1,) * remainder
    input_dims = tuple(inputs.shape[ax] for ax in axis)
    kernel_shape = input_dims + features
    kernel = self.param('kernel', kernel_init_wrap, batch_shape + kernel_shape,
                        self.param_dtype)
    alpha = self.param('fast_weight_alpha',
                       alpha_init_wrap,
                       batch_shape + (self.ens_size,) + input_dims,
                       self.param_dtype)
    gamma = self.param('fast_weight_gamma',
                       gamma_init_wrap,
                       batch_shape + (self.ens_size,) + features,
                       self.param_dtype)

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_batch_dims + n_axis))

    if self.use_bias:
      def bias_init_wrap(rng, shape, dtype=jnp.float32):
        size_batch_dims = np.prod(shape[:n_batch_dims], dtype=np.int32)
        flat_shape = (self.ens_size, np.prod(shape[-n_features:]))
        bias = jnp.concatenate([self.bias_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
        return jnp.reshape(bias, shape)

      bias = self.param('bias',
                        bias_init_wrap,
                        batch_shape + (self.ens_size,) + features,
                        self.param_dtype)
    else:
      bias = None

    inputs, kernel, alpha, gamma, bias = nn.dtypes.promote_dtype(
        inputs, kernel, alpha, gamma, bias, dtype=self.dtype)

    alpha = jnp.reshape(alpha, expanded_batch_shape + input_dims)
    inputs *= alpha
    out = lax.dot_general(inputs,
                          kernel,
                          ((axis, contract_ind), (batch_dims, batch_ind)),
                          precision=self.precision)
    gamma = jnp.reshape(gamma, expanded_batch_shape + features)
    out *= gamma
    # dot_general output has shape batch_dims + [ens_size] + feature_dims
    if self.use_bias:
      # expand bias shape to broadcast bias over batch dims.
      bias = jnp.reshape(bias, expanded_batch_shape + features)
      out += bias
    return out


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
  dtype: Optional[Dtype] = None
  alpha_init: InitializeFn = nn.initializers.ones
  gamma_init: InitializeFn = nn.initializers.ones
  kernel_init: InitializeFn = default_kernel_init
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

    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.ens_size, self.features),
                        dtype)
      bias_shape = (self.ens_size,) + (1,) * (outputs.ndim - 2) + (
          self.features,)
      outputs = outputs + jnp.reshape(bias, bias_shape)

    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    return jnp.reshape(outputs, inputs_shape[:-1] + (self.features,))
