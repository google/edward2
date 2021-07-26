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

"""Definitions for random feature Gaussian process layer.

## References:

[1]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
     Machines. In _Neural Information Processing Systems_, 2007.
     https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
"""
import dataclasses
from typing import Any, Callable, Iterable, Optional, Union

import flax.linen as nn

from jax import lax
from jax import random
import jax.numpy as jnp

# Jax-related data types.
PRNGKey = Any
Shape = Iterable[int]
Dtype = type(jnp.float32)
Array = jnp.ndarray
Initializer = Callable[[PRNGKey, Shape, Dtype], Array]

# Default config for random features.
default_rbf_activation = jnp.cos
default_rbf_kernel_init = nn.initializers.normal(stddev=1.)
default_rbf_bias_init = nn.initializers.uniform(scale=2. * jnp.pi)

# Default field value for kwargs, to be used for data class declaration.
default_kwarg_dict = lambda: dataclasses.field(default_factory=dict)


class RandomFourierFeatures(nn.Module):
  """A random fourier feature (RFF) layer that approximates a kernel model.

  The random feature transformation is a one-hidden-layer network with
  non-trainable weights (see, e.g., Algorithm 1 of [1]). Specifically:

  f(x) = activation(x @ kernel + bias) * output_scale.

  The forward pass logic closely follows that of the nn.Dense.

  Attributes:
    features: the number of output units.
    feature_scalefeature_scale: scale to apply to the output
      (default: sqrt(2. / features), see Algorithm 1 of [1]).
    activation: activation function to apply to the output.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    seed: random seed for generating random features (default: 0). This will
      override the external RNGs.
    dtype: the dtype of the computation (default: float32).
  """
  features: int
  feature_scale: Optional[jnp.float32] = None
  activation: Callable[[Array], Array] = default_rbf_activation
  kernel_init: Initializer = default_rbf_kernel_init
  bias_init: Initializer = default_rbf_bias_init
  seed: int = 0
  dtype: Dtype = jnp.float32
  collection_name: str = 'random_features'

  def setup(self):
    # Defines the random number generator.
    self.rng = random.PRNGKey(self.seed)

    # Processes random feature scale.
    self._feature_scale = self.feature_scale
    if self._feature_scale is None:
      self._feature_scale = jnp.sqrt(2. / self.features)
    self._feature_scale = jnp.asarray(self._feature_scale, dtype=self.dtype)

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies random feature transformation along the last dimension of inputs.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    # Initializes variables.
    input_dim = inputs.shape[-1]
    kernel, bias = self.initialize_random_features(input_dim)

    # Specifies multiplication dimension.
    contracting_dims = ((inputs.ndim - 1,), (0,))
    batch_dims = ((), ())

    # Performs forward pass.
    inputs = jnp.asarray(inputs, self.dtype)
    outputs = lax.dot_general(inputs, kernel.value,
                              (contracting_dims, batch_dims))
    outputs = outputs + jnp.broadcast_to(bias.value, outputs.shape)

    return self._feature_scale * self.activation(outputs)

  def initialize_random_features(
      self, input_dim: int) -> Union[nn.Variable, nn.Variable]:
    """Initialize the untrainable kernel and bias."""
    kernel_shape = (input_dim, self.features)
    kernel_rng, bias_rng = random.split(self.rng, num=2)

    # Initialization random feature values.
    kernel_val = self.kernel_init(kernel_rng, kernel_shape, self.dtype)
    bias_val = self.bias_init(bias_rng, (self.features,), self.dtype)

    # Assign to variables.
    kernel = self.variable(self.collection_name, 'kernel', lambda: kernel_val)
    bias = self.variable(self.collection_name, 'bias', lambda: bias_val)

    return kernel, bias
