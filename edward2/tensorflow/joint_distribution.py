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

"""The `JointDistributionCoroutine` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from edward2 import tracers
from edward2 import trace

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointDistributionEdward2',
]


def make_sequential_value_setter(*model_args):
  model_args = list(model_args)
  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    kwargs['value'] = model_args.pop(0)
    return trace.traceable(f)(*args, **kwargs)
  return set_values


@contextlib.contextmanager
def sequential_tape():
  # Records all RVs, even those without names.
  tape_data = []
  def record(f, *args, **kwargs):
    """Records execution to a tape."""
    output = trace.traceable(f)(*args, **kwargs)
    tape_data.append(output)
    return output
  with trace.trace(record):
    yield tape_data


def _intercept_to_set_values(fn, value):
  def intercepted_fn():
    with trace.trace(make_sequential_value_setter(*value)):
      fn()
  return intercepted_fn


def _intercept_to_set_seed(fn, seed):
  def intercepted_fn():
    with trace.trace(tracers.set_seed(seed)):
      fn()
  return intercepted_fn


class JointDistributionEdward2(
    tfp.distributions.AutovectorizedJointDistribution):
  """Joint distribution parameterized by an Edward2 model.

  This distribution enables both sampling and joint probability computation from
  a single model specification.


  #### Examples


  #### Discussion


  """

  def __init__(self,
               model,
               sample_dtype=None,
               validate_args=False,
               name=None):
    """Construct the `JointDistributionEdward2` distribution.

    Args:
      model: A Python callable representing an Edward2 model. The callable
        takes zero arguments and its return value is ignored; it specifies
        a model by internally creating interceptable Edward2 `RandomVariable`s.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `"JointDistributionEdward2"`).
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'JointDistributionEdward2') as name:
      self._model = model
      self._sampled_distributions = None
      self._bare_samples = None
      self._sample_dtype = sample_dtype
      super(JointDistributionEdward2, self).__init__(
          reparameterization_type=None,  # Ignored; we'll override.
          dtype=sample_dtype,  # TODO(davmre): do we need this?
          validate_args=validate_args,
          allow_nan_stats=False,
          parameters=parameters,
          graph_parents=[],
          name=name)

  def _flat_sample_distributions(self, sample_shape=(), seed=None, value=None):
    """Executes `model`, creating both samples and distributions."""

    if sample_shape:
      raise ValueError('sample shape not supported!')
    del sample_shape

    model = self._model
    if value is not None:
      model = _intercept_to_set_values(model, value)
    if seed is not None:
      model = _intercept_to_set_seed(model, seed)

    with sequential_tape() as model_tape:
      model()

    if model_tape:
      ds, xs = zip(*[(rv.distribution, rv.value) for rv in model_tape])
    else:
      ds, xs = [], []
    return ds, xs

  def _model_unflatten(self, xs):
    if self._sample_dtype is None:
      return tuple(xs)
    # Cast `xs` as `tuple` so we can handle generators.
    return tf.nest.pack_sequence_as(self._sample_dtype, tuple(xs))

  def _model_flatten(self, xs):
    if self._sample_dtype is None:
      return tuple(xs)
    return nest.flatten_up_to(self._sample_dtype, xs)
