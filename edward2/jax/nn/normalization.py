# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Normalization layers.

## References:

[1] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.

[2] Farzan Farnia, Jesse M. Zhang, David Tse.
    Generalizable Adversarial Training via Spectral Normalization.
    In _International Conference on Learning Representations_, 2019.

[3] Stephan Hoyer.
    Better spectral norm for convolutions.
    https://nbviewer.jupyter.org/gist/shoyer/fa9a29fd0880e2e033d7696585978bfc
"""

from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import flax.core
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

# conventional Flax types
Axes = Union[int, Iterable[int]]
Array = Any
Dtype = Any
PRNGKey = Any
Shape = Tuple[int]


def _l2_normalize(x, eps=1e-12):
  return x * jax.lax.rsqrt(jnp.maximum(jnp.square(x).sum(), eps))


class SpectralNormalization(nn.Module):
  """Implements spectral normalization for linear layers.

  In Flax, parameters are immutable so we cannot modify the parameters of
  the input layer during the transformation. As a resolution, we will move
  all parameters of the input layer to this spectral normalization layer.
  During the transformation, we will modify the weight and call the input
  layer with the updated weight.
  For example, the pattern for parameters with a Dense input layer will be
    {"Dense": {"weight": ..., "bias": ...}}
  which matches the pattern in Flax as if "Dense" is a submodule of this
  spectral normalization layer. Name of the input layer can be customized
  using the `layer_name` attribute.

  Note that currently, the implementation with a customized
  ``kernel_apply_kwargs`` attribute only works for Flax's Dense, Conv, Embed
  (or their subclasses) input layers.

  Attributes:
    layer: a Flax layer to apply normalization to.
    iteration: the number of power iterations to estimate weight matrix's
      singular value.
    norm_multiplier: multiplicative constant to threshold the normalization.
      Usually under normalization, the singular value will converge to this
      value.
    u_init: initializer function for the first left singular vectors of the
      kernel.
    v_init: initializer function for the first right singular vectors of the
      kernel.
    kernel_apply_kwargs: updated keyword arguments to clone the input layer. The
      cloned layer represents the linear operator performed by the weight
      matrix. If not specified, that operator follows SN-GAN implementation
      (reference [1]). In particular, for Dense layers the default behavior is
      equivalent to using a cloned layer with no bias (by specifying
      ``kernel_apply_kwargs=dict(use_bias=False)``). With this customization, we
      can have the same implementation (inspried by reference [3]) for different
      interpretations of Conv layers. Also see SpectralNormalizationConv2D for
      an example of using this attribute.
    kernel_name: name of the kernel parameter of the input layer.
    layer_name: name of the input layer.
  """
  layer: nn.Module
  iteration: int = 1
  norm_multiplier: float = 0.95
  u_init: Callable[[PRNGKey, Shape, Dtype],
                   Array] = nn.initializers.normal(stddev=0.05)
  v_init: Callable[[PRNGKey, Shape, Dtype],
                   Array] = nn.initializers.normal(stddev=0.05)
  kernel_apply_kwargs: Optional[Mapping[str, Any]] = None
  # TODO(phandu): Allow users to provide a list of kernel names, so that
  # we can use SN-GAN interpretation for SeparableConv2D layers.
  kernel_name: str = "kernel"
  layer_name: Optional[str] = None

  def _get_singular_vectors(self, initializing, kernel_apply, in_shape, dtype):
    if initializing:
      rng_u = self.make_rng("params")
      rng_v = self.make_rng("params")
      # Interpret output shape (not that this does not cost any FLOPs).
      out_shape = jax.eval_shape(kernel_apply,
                                 jax.ShapeDtypeStruct(in_shape, dtype)).shape
    else:
      rng_u = rng_v = out_shape = None
    u = self.variable("spectral_stats", "u", self.u_init, rng_u, out_shape,
                      dtype)
    v = self.variable("spectral_stats", "v", self.v_init, rng_v, in_shape,
                      dtype)
    return u, v

  @nn.compact
  def __call__(self, inputs: Array, training: bool = True) -> Array:
    """Applies a linear transformation with spectral normalization to the inputs.

    Args:
      inputs: The nd-array to be transformed.
      training: Whether to perform power interations to update the singular
        value estimate.

    Returns:
      The transformed input.
    """
    layer_name = type(
        self.layer).__name__ if self.layer_name is None else self.layer_name
    params = self.param(layer_name,
                        lambda *args: self.layer.init(*args)["params"], inputs)
    w = params[self.kernel_name]

    if self.kernel_apply_kwargs is None:
      # By default, we use the implementation in SN-GAN.
      kernel_apply = lambda x: x @ w.reshape(-1, w.shape[-1])
      in_shape = (np.prod(w.shape[:-1]),)
    else:
      # Otherwise, we extract the actual kernel transformation in the input
      # layer. This is useful for Conv2D spectral normalization in [2].
      kernel_apply = self.layer.clone(**self.kernel_apply_kwargs).bind(  # pylint: disable=not-a-mapping
          {"params": {
              self.kernel_name: w
          }})
      # Compute input shape of the kernel operator. This is correct for all
      # linear layers on Flax: Dense, Conv, Embed.
      in_shape = inputs.shape[-w.ndim + 1:-1] + w.shape[-2:-1]

    initializing = self.is_mutable_collection("params")
    u, v = self._get_singular_vectors(initializing, kernel_apply, in_shape,
                                      w.dtype)
    u_hat, v_hat = u.value, v.value
    u_, kernel_transpose = jax.vjp(kernel_apply, v_hat)
    if training and not initializing:
      # Run power iterations using autodiff approach inspired by [3].
      def scan_body(carry, _):
        u_hat, v_hat, u_ = carry
        v_, = kernel_transpose(u_hat)
        v_hat = _l2_normalize(v_)
        u_ = kernel_apply(v_hat)
        u_hat = _l2_normalize(u_)
        return (u_hat, v_hat, u_), None

      (u_hat, v_hat, u_), _ = jax.lax.scan(
          scan_body, (u_hat, v_hat, u_), None, length=self.iteration)
      u.value, v.value = u_hat, v_hat

    sigma = jnp.vdot(u_hat, u_)
    # Bound spectral norm by the `norm_multiplier`.
    sigma = jnp.maximum(sigma / self.norm_multiplier, 1.)
    w_hat = w / jax.lax.stop_gradient(sigma)
    self.sow("intermediates", "w", w_hat)

    # Update params.
    params = flax.core.unfreeze(params)
    params[self.kernel_name] = w_hat
    layer_params = flax.core.freeze({"params": params})
    return self.layer.apply(layer_params, inputs)


class SpectralNormalizationConv2D(SpectralNormalization):
  __doc__ = "Implements spectral normalization for Conv layers based on [2].\n" + "\n".join(
      SpectralNormalization.__doc__.split("\n")[1:])

  kernel_apply_kwargs: Mapping[str, Any] = flax.core.FrozenDict(
      feature_group_count=1, padding="SAME", use_bias=False)


# Taken from
# https://github.com/google/flax/blob/main/flax/linen/normalization.py
def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None):
  """Computes mean and variance statistics."""
  def _abs_sq(x):
    """Computes the elementwise square of the absolute value |x|^2."""
    if jnp.iscomplexobj(x):
      return lax.square(lax.real(x)) + lax.square(lax.imag(x))
    else:
      return lax.square(x)
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  mean = jnp.mean(x, axes)
  mean2 = jnp.mean(_abs_sq(x), axes)
  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var = jnp.maximum(0., mean2 - _abs_sq(mean))
  return mean, var


# Adapted from
# https://github.com/google/flax/blob/main/flax/linen/normalization.py
def _normalize_ensemble(
    mdl: nn.Module, x: Array, mean: Array, var: Array,
    reduction_axes: Axes, feature_axes: Axes,
    dtype: Dtype, param_dtype: Dtype,
    epsilon: float,
    use_bias: bool, use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
    ens_size: int):
  """"Normalizes input per ensemble member with learnable biases and scales."""
  def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
    """Returns a tuple of deduplicated, sorted, and positive axes."""
    if not isinstance(axes, Iterable):
      axes = (axes,)
    return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  # We broadcast operations by reshaping ens_size as the first outer axis.
  feature_shape = [1] * x.ndim
  reduced_feature_shape = [ens_size]
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  feature_shape = [ens_size] + feature_shape
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  y = jnp.reshape(y, (ens_size, -1) + y.shape[1:])
  mul = jnp.reshape(mul, (ens_size, -1) + mul.shape[1:])
  if use_scale:
    scale = mdl.param("scale", scale_init, reduced_feature_shape,
                      param_dtype).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = mdl.param("bias", bias_init, reduced_feature_shape,
                     param_dtype).reshape(feature_shape)
    y += bias
  y = jnp.reshape(y, (-1,) + y.shape[2:])
  return jnp.asarray(y, dtype)


class LayerNormEnsemble(nn.Module):
  """Layer normalization applied separately to each ensemble member.

  The implementation is vectorized so that most computation is shared (as it is
  mathematically) and the shift and scale transforms are parallelized across
  the ensemble dimension.

  Attributes:
    ens_size: Ensemble size.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  If True, bias (beta) is added.
    use_scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.
  """
  ens_size: int
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  param_dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.ones

  @nn.compact
  def __call__(self, x):
    """Applies layer normalization on the input.

    Args:
      x: Inputs of shape [batch * ens_size, ..., hidden_size].

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    reduction_axes = (-1,)
    feature_axes = (-1,)
    mean, var = _compute_stats(x, reduction_axes, None, None)
    return _normalize_ensemble(
        self, x, mean, var, reduction_axes, feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init,
        ens_size=self.ens_size)
