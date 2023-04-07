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

"""JAX layer and utils."""

from typing import Callable, Iterable, Optional

from jax import random
import jax.numpy as jnp

Array = jnp.ndarray
DType = type(jnp.float32)
InitializeFn = Callable[[jnp.ndarray, Iterable[int], DType], jnp.ndarray]


def make_sign_initializer(random_sign_init: float) -> InitializeFn:
  """Builds initializer with specified random_sign_init.

  Args:
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.

  Returns:
    nn.initializers
  """
  if random_sign_init > 0:
    def initializer(key, shape, dtype=jnp.float32):
      x = 2 * random.bernoulli(key, random_sign_init, shape) - 1.0
      return x.astype(dtype)
    return initializer
  else:
    def initializer(key, shape, dtype=jnp.float32):
      x = random.normal(key, shape, dtype) * (-random_sign_init) + 1.0
      return x.astype(dtype)
    return initializer


def mean_field_logits(logits: Array,
                      covmat: Optional[Array] = None,
                      mean_field_factor: float = 1.,
                      likelihood: str = 'logistic'):
  """Adjust the model logits so its softmax approximates the posterior mean [4].

  Arguments:
    logits: A float ndarray of shape (batch_size, num_classes).
    covmat: A float ndarray of shape (batch_size, ). If None then it is assumed
      to be a vector of 1.'s.
    mean_field_factor: The scale factor for mean-field approximation, used to
      adjust the influence of posterior variance in posterior mean
      approximation. If covmat=None then it is used as the scaling parameter for
      temperature scaling.
    likelihood: name of the likelihood for integration in Gaussian-approximated
      latent posterior. Must be one of ('logistic', 'binary_logistic',
      'poisson').

  Returns:
    A float ndarray of uncertainty-adjusted logits, shape
    (batch_size, num_classes).

  Raises:
    (ValueError) If likelihood is not one of ('logistic', 'binary_logistic',
    'poisson').
  """
  if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
    raise ValueError(
        f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
    )

  if mean_field_factor < 0:
    return logits

  # Defines predictive variance.
  variances = 1. if covmat is None else covmat

  # Computes scaling coefficient for mean-field approximation.
  if likelihood == 'poisson':
    logits_scale = jnp.exp(-variances * mean_field_factor / 2.)  # pylint:disable=invalid-unary-operand-type
  else:
    logits_scale = jnp.sqrt(1. + variances * mean_field_factor)

  # Pads logits_scale to compatible dimension.
  while logits_scale.ndim < logits.ndim:
    logits_scale = jnp.expand_dims(logits_scale, axis=-1)

  return logits / logits_scale


