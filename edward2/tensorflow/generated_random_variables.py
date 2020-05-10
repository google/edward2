# coding=utf-8
# Copyright 2020 The Edward2 Authors.
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

"""Automatically generated random variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import re

from edward2.tensorflow.random_variable import RandomVariable
from edward2.trace import traceable
import six
import tensorflow_probability as tfp


def expand_docstring(**kwargs):
  """Decorator to programmatically expand the docstring.

  Args:
    **kwargs: Keyword arguments to set. For each key-value pair `k` and `v`,
      the key is found as `${k}` in the docstring and replaced with `v`.

  Returns:
    Decorated function.
  """
  def _fn_wrapped(fn):
    """Original function with modified `__doc__` attribute."""
    doc = inspect.cleandoc(fn.__doc__)
    for k, v in six.iteritems(kwargs):
      # Capture each ${k} reference to replace with v.
      # We wrap the replacement in a function so no backslash escapes
      # are processed.
      pattern = r"\$\{" + str(k) + r"\}"
      doc = re.sub(pattern, lambda match: v, doc)  # pylint: disable=cell-var-from-loop
    fn.__doc__ = doc
    return fn
  return _fn_wrapped


def make_random_variable(distribution_cls):
  """Factory function to make random variable given distribution class."""
  @traceable
  @functools.wraps(distribution_cls, assigned=("__module__", "__name__"))
  @expand_docstring(cls=distribution_cls.__name__,
                    doc=inspect.cleandoc(
                        distribution_cls.__init__.__doc__ if
                        distribution_cls.__init__.__doc__ is not None else ""))
  def func(*args, **kwargs):
    # pylint: disable=g-doc-args
    """Create a random variable for ${cls}.

    See ${cls} for more details.

    Returns:
      RandomVariable.

    #### Original Docstring for Distribution

    ${doc}
    """
    # pylint: enable=g-doc-args
    sample_shape = kwargs.pop("sample_shape", ())
    value = kwargs.pop("value", None)
    return RandomVariable(distribution=distribution_cls(*args, **kwargs),
                          sample_shape=sample_shape,
                          value=value)
  return func


__all__ = ["make_random_variable"]
_globals = globals()
for candidate_name in sorted(dir(tfp.distributions)):
  candidate = getattr(tfp.distributions, candidate_name)
  if (inspect.isclass(candidate) and
      candidate != tfp.distributions.Distribution and
      issubclass(candidate, tfp.distributions.Distribution)):

    _globals[candidate_name] = make_random_variable(candidate)
    __all__.append(candidate_name)

_HAS_DYNAMIC_ATTRIBUTES = True
