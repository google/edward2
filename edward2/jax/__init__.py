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

"""Edward2 probabilistic programming language with JAX backend."""

from edward2.jax import nn
from edward2.maps import robust_map
from edward2.trace import get_next_tracer
from edward2.trace import trace
from edward2.trace import traceable
from edward2.tracers import condition
from edward2.tracers import tape
from edward2.version import __version__
from edward2.version import VERSION

__all__ = [
    "condition",
    "get_next_tracer",
    "nn",
    "robust_map",
    "tape",
    "trace",
    "traceable",
    "__version__",
    "VERSION",
]
