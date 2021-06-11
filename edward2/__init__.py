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

"""Edward2 probabilistic programming language.

For user guides, see:

+ [Overview](
   https://github.com/google/edward2/blob/master/README.md)
+ [Upgrading from Edward to Edward2](
   https://github.com/google/edward2/blob/master/Upgrading_From_Edward_To_Edward2.md)

"""

import warnings

__all__ = []
try:
  from edward2 import jax
  __all__ += ["jax"]
except ImportError:
  warnings.warn("JAX backend for Edward2 is not available.")

try:
  from edward2 import numpy
  __all__ += ["numpy"]
except ImportError:
  warnings.warn("NumPy backend for Edward2 is not available.")

try:
  from edward2 import tensorflow
  from edward2.tensorflow import *  # pylint: disable=wildcard-import
  # By default, `import edward2 as ed` uses the TensorFlow backend's namespace.
  __all__ += tensorflow.__all__ + ["tensorflow"]
except ImportError:
  warnings.warn("TensorFlow backend for Edward2 is not available.")


if all(x not in __all__ for x in ("jax", "numpy", "tensorflow")):
  raise ImportError("No backend is available for Edward2. Please install "
                    "dependencies for JAX, NumPy, or TensorFlow.")
# pylint: enable=g-import-not-at-top
