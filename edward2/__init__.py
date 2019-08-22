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

"""Edward2 probabilistic programming language.

For user guides, see:

+ [Overview](
   https://github.com/google/edward2/blob/master/README.md)
+ [Upgrading from Edward to Edward2](
   https://github.com/google/edward2/blob/master/Upgrading_From_Edward_To_Edward2.md)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
from edward2 import numpy
from edward2 import tensorflow
from edward2.tensorflow import *  # pylint: disable=wildcard-import

_allowed_symbols = [
    "numpy",
    "tensorflow",
]
# By default, `import edward2 as ed` uses the TensorFlow backend's namespace.
for name in dir(tensorflow):
  _allowed_symbols.append(name)

try:
  from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top
except ImportError:
  __all__ = _allowed_symbols
  try:
    import numpy as np  # pylint: disable=g-import-not-at-top,unused-import
  except ImportError:
    warnings.warn("Neither NumPy nor TensorFlow backends are available for "
                  "Edward2.")
else:
  remove_undocumented(__name__, _allowed_symbols)
