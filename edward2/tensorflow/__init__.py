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

"""Edward2 probabilistic programming language with TensorFlow backend."""
asdf
Hello world!

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow.program_transformations import make_log_joint_fn
from edward2.tensorflow.random_variable import RandomVariable
from edward2.trace import get_next_tracer
from edward2.trace import trace
from edward2.trace import traceable
from edward2.tracers import condition
from edward2.tracers import tape
from edward2.version import __version__
from edward2.version import VERSION

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import


_allowed_symbols = [
    "RandomVariable",
    "condition",
    "get_next_tracer",
    "make_log_joint_fn",
    "make_random_variable",
    "tape",
    "trace",
    "traceable",
    "__version__",
    "VERSION",
]
# Make the TensorFlow backend be optional without mandatory dependencies.
try:
  # pylint: disable=g-import-not-at-top
  from tensorflow_probability import distributions
  from edward2.tensorflow import generated_random_variables
  from edward2.tensorflow.generated_random_variables import *  # pylint: disable=wildcard-import
  from edward2.tensorflow.generated_random_variables import make_random_variable
  # pylint: enable=g-import-not-at-top
  for name in dir(generated_random_variables):
    if name in sorted(dir(distributions)):
      _allowed_symbols.append(name)
except ImportError:
  pass

remove_undocumented(__name__, _allowed_symbols)
