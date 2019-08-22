# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

"""Edward2 probabilistic programming language with NumPy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Make the NumPy backend be optional. The namespace is empty if NumPy
# is not available.
# pylint: disable=g-import-not-at-top
try:
  import numpy as np  # pylint: disable=unused-import
  from scipy import stats
except ImportError:
  pass
else:
  from edward2.numpy import generated_random_variables
  from edward2.numpy.generated_random_variables import *  # pylint: disable=wildcard-import
  from edward2.numpy.program_transformations import make_log_joint_fn
  from edward2.trace import get_next_tracer
  from edward2.trace import trace
  from edward2.trace import traceable
  from edward2.tracers import condition
  from edward2.tracers import tape
  from edward2.version import __version__
  from edward2.version import VERSION

  _allowed_symbols = [
      "condition",
      "get_next_tracer",
      "make_log_joint_fn",
      "tape",
      "trace",
      "traceable",
      "__version__",
      "VERSION",
  ]
  for name in dir(generated_random_variables):
    if name in sorted(dir(stats)):
      _allowed_symbols.append(name)

  try:
    from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import
  except ImportError:
    __all__ = _allowed_symbols
  else:
    remove_undocumented(__name__, _allowed_symbols)
# pylint: enable=g-import-not-at-top
