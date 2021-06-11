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

"""JAX layer and utils."""

from typing import Iterable, Callable

from jax import random
import jax.numpy as jnp

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
