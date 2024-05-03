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

"""Random variable."""

import functools
import tensorflow as tf


# TODO(b/202447704): Update to use public Protocol when available.
class RandomVariableTraceType:
  """Class outlining Tracing Protocol for RandomVariable."""

  def __init__(self, instance_id):
    self.instance_id = instance_id

  def is_subtype_of(self, other):
    return self == other

  def most_specific_common_supertype(self, others):
    if not all(self == other for other in others):
      return None

    return self

  def __hash__(self) -> int:
    return self.instance_id

  def __eq__(self, other) -> bool:
    if not isinstance(other, RandomVariableTraceType):
      return False

    return self.instance_id == other.instance_id

  def __repr__(self):
    return f"RandomVariableTraceType(instance_id={self.instance_id})"


class RandomVariable(object):
  """Class for random variables.

  `RandomVariable` encapsulates properties of a random variable, namely, its
  distribution, sample shape, and (optionally overridden) value. Its `value`
  property is a `tf.Tensor`, which embeds the `RandomVariable` object into the
  TensorFlow graph. `RandomVariable` also features operator overloading,
  enabling idiomatic usage as if one were operating on `tf.Tensor`s.

  The random variable's shape is given by

  `sample_shape + distribution.batch_shape + distribution.event_shape`,

  where `sample_shape` is an optional argument describing the shape of
  independent, identical draws from the distribution (default is `()`, meaning
  a single draw); `distribution.batch_shape` describes the shape of
  independent-but-not-identical draws (determined by the shape of the
  distribution's parameters); and `distribution.event_shape` describes the
  shape of dependent dimensions (e.g., `Normal` has scalar `event_shape`;
  `Dirichlet` has vector `event_shape`).

  #### Examples

  ```python
  import edward2 as ed
  import tensorflow_probability as tfp

  z1 = tf.constant([[1.0, -0.8], [0.3, -1.0]])
  z2 = tf.constant([[0.9, 0.2], [2.0, -0.1]])
  x = ed.RandomVariable(tfp.distributions.Bernoulli(logits=tf.matmul(z1, z2)))

  loc = ed.RandomVariable(tfp.distributions.Normal(0., 1.))
  x = ed.RandomVariable(tfp.distributions.Normal(loc, 1.), sample_shape=50)
  assert x.shape.as_list() == [50]
  assert x.sample_shape.as_list() == [50]
  assert x.distribution.batch_shape.as_list() == []
  assert x.distribution.event_shape.as_list() == []
  ```
  """

  def __init__(self,
               distribution,
               sample_shape=(),
               value=None):
    """Create a new random variable.

    Args:
      distribution: Distribution of the random variable. At minimum, the
        distribution object must have the attributes `dtype`, `batch_shape`,
        `event_shape`, `sample`, and `name`.
      sample_shape: tf.TensorShape of samples to draw from the random variable.
        Default is `()` corresponding to a single sample.
      value: tf.Tensor to associate with random variable. Must have shape
        `sample_shape + distribution.batch_shape + distribution.event_shape`.
        Default is to sample from random variable according to `sample_shape`.

    Raises:
      ValueError: `value` has incompatible shape with
        `sample_shape + distribution.batch_shape + distribution.event_shape`.
    """
    self._distribution = distribution
    self._sample_shape = sample_shape
    if tf.is_tensor(value):
      value_shape = value.shape
      expected_value_shape = self.sample_shape.concatenate(
          self.distribution.batch_shape).concatenate(
              self.distribution.event_shape)
      if not value_shape.is_compatible_with(expected_value_shape):
        raise ValueError(
            "Incompatible shape for initialization argument 'value'. "
            "Expected %s, got %s." % (expected_value_shape, value_shape))
    self._value = value
    self._value_shape = self.value.shape

  @property
  def distribution(self):
    """Distribution of random variable."""
    return self._distribution

  @property
  def dtype(self):
    """`Dtype` of elements in this random variable."""
    return self.value.dtype

  @property
  def sample_shape(self):
    """Sample shape of random variable as a `TensorShape`."""
    if tf.is_tensor(self._sample_shape):
      return tf.TensorShape(tf.get_static_value(self._sample_shape))
    return tf.TensorShape(self._sample_shape)

  def sample_shape_tensor(self, name="sample_shape_tensor"):
    """Sample shape of random variable as a 1-D `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      sample_shape: `Tensor`.
    """
    with tf.name_scope(name):
      if tf.is_tensor(self._sample_shape):
        return self._sample_shape
      return tf.convert_to_tensor(self.sample_shape.as_list(), dtype=tf.int32)

  @property
  def shape(self):
    """Shape of random variable."""
    return self._value_shape

  @property
  def value(self):
    """Get tensor that the random variable corresponds to."""
    if self._value is None:
      try:
        self._value = self.distribution.sample(self.sample_shape_tensor())
      except NotImplementedError:
        raise NotImplementedError(
            "sample is not implemented for {0}. You must either pass in the "
            "value argument or implement sample for {0}."
            .format(self.distribution.__class__.__name__))
    else:
      self._value = tf.cast(self._value, self.distribution.dtype)
    return self._value

  def __str__(self):
    name = _numpy_text(self.value)
    return "RandomVariable(\"%s\"%s%s%s)" % (
        name,
        ", shape=%s" % self.shape if self.shape.ndims is not None else "",
        ", dtype=%s" % self.dtype.name if self.dtype else "",
        ", device=%s" % self.value.device if self.value.device else "")

  def __repr__(self):
    string = "ed.RandomVariable '%s' shape=%s dtype=%s" % (
        self.distribution.name, self.shape, self.dtype.name)
    if hasattr(self.value, "numpy"):
      string += " numpy=%s" % _numpy_text(self.value, is_repr=True)
    return "<%s>" % string

  def __getitem__(self, slices):
    value = self.value.__getitem__(slices)
    if self.sample_shape.as_list():
      # Return an indexed Tensor instead of RandomVariable if sample_shape is
      # non-scalar. Sample shapes can complicate how to index the distribution.
      return value
    try:
      distribution = self.distribution.__getitem__(slices)
    except (tf.errors.InvalidArgumentError, ValueError):
      return value
    else:
      return RandomVariable(distribution, value=value)

  def __hash__(self):
    return id(self)

  def __eq__(self, other):
    return id(self) == id(other)

  def __ne__(self, other):
    return not self == other

  def numpy(self):
    """Value as NumPy array."""
    return self.value.numpy()

  def get_shape(self):
    """Get shape of random variable."""
    return self.shape

  def __tf_tracing_type__(self, _):
    return RandomVariableTraceType(id(self))

  # This enables the RandomVariable's overloaded "right" binary operators to
  # run when the left operand is an ndarray, because it accords the
  # RandomVariable class higher priority than an ndarray, or a numpy matrix.
  __array_priority__ = 100


def _numpy_text(tensor, is_repr=False):
  """Human-readable representation of a tensor's numpy value."""
  if tensor.dtype.is_numpy_compatible:
    text = repr(tensor.numpy()) if is_repr else str(tensor.numpy())
  else:
    text = "<unprintable>"
  if "\n" in text:
    text = "\n" + text
  return text


def _overload_operator(cls, op):
  """Defer an operator overload to `tf.Tensor`.

  We pull the operator out of tf.Tensor dynamically to avoid ordering issues.

  Args:
    cls: Class to overload operator.
    op: Python string representing the operator name.
  """
  @functools.wraps(getattr(tf.Tensor, op))
  def _run_op(a, *args):
    return getattr(tf.Tensor, op)(a.value, *args)

  setattr(cls, op, _run_op)


def _tensor_conversion_function(v, dtype=None, name=None, as_ref=False):
  del name, as_ref  # unused
  if dtype and not dtype.is_compatible_with(v.dtype):
    raise ValueError(
        "Incompatible type conversion requested to type '%s' for variable "
        "of type '%s'" % (dtype.name, v.dtype.name))
  return v.value


for operator in tf.Tensor.OVERLOADABLE_OPERATORS.difference(
    {"__getitem__"}).union({"__iter__", "__bool__", "__nonzero__"}):
  _overload_operator(RandomVariable, operator)

tf.register_tensor_conversion_function(  # enable tf.convert_to_tensor
    RandomVariable, _tensor_conversion_function)
