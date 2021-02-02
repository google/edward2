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

"""Reversible layers."""

from edward2.tensorflow import random_variable
from edward2.tensorflow import transformed_random_variable
from edward2.tensorflow.layers import utils
import tensorflow as tf


# TODO(trandustin): Move Reverse to another module(?).
class Reverse(tf.python.keras.layers.Layer):
  """Swaps the forward and reverse transformations of a layer."""

  def __init__(self, reversible_layer, **kwargs):
    super(Reverse, self).__init__(**kwargs)
    if not hasattr(reversible_layer, 'reverse'):
      raise ValueError('Layer passed-in has not implemented "reverse" method: '
                       '{}'.format(reversible_layer))
    self.call = reversible_layer.reverse
    self.reverse = reversible_layer.call


class DiscreteAutoregressiveFlow(tf.python.keras.layers.Layer):
  """A discrete reversible layer.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)

  For the forward pass, the flow computes in serial:

  ```none
  outputs = []
  for t in range(length):
    new_inputs = [outputs, inputs[..., t, :]]
    net = layer(new_inputs)
    loc, scale = tf.split(net, 2, axis=-1)
    loc = tf.argmax(loc, axis=-1)
    scale = tf.argmax(scale, axis=-1)
    new_outputs = (((inputs - loc) * inverse(scale)) % vocab_size)[..., -1, :]
    outputs.append(new_outputs)
  ```

  For the reverse pass, the flow computes in parallel:

  ```none
  net = layer(inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = (loc + scale * inputs) % vocab_size
  ```

  The modular arithmetic happens in one-hot space.

  If `x` is a discrete random variable, the induced probability mass function on
  the outputs `y = flow(x)` is

  ```none
  p(y) = p(flow.reverse(y)).
  ```

  The location-only transform is always invertible ([integers modulo
  `vocab_size` form an additive group](
  https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
  is invertible if the scale and `vocab_size` are coprime (see
  [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
  """

  def __init__(self, layer, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(DiscreteAutoregressiveFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2]
    if length is None:
      raise NotImplementedError('length dimension must be known.')
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    # TODO(trandustin): Use tf.while_loop. Unrolling is memory-expensive for big
    # models and not valid for variable lengths.
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].

    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs[..., tf.newaxis, :]
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, paddings=[[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = loc[..., 0:1, :]
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = scale[..., 0:1, :]
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      inverse_scale = utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = utils.one_hot_minus(inputs, loc)
      outputs = utils.one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., 0:1, :]
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      outputs = utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].

    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs,
        paddings=[[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    net = self.layer(padded_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = scale[..., :(timestep+1), :]
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      inverse_scale = utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = utils.one_hot_minus(inputs, loc)
      new_outputs = utils.one_hot_multiply(shifted_inputs, inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = loc[..., :(timestep+1), :]
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      new_outputs = utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    net = self.layer(inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      scaled_inputs = utils.one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    outputs = utils.one_hot_add(loc, scaled_inputs)
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


class DiscreteBipartiteFlow(tf.python.keras.layers.Layer):
  """A discrete reversible layer.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)

  For the forward pass, the flow computes:

  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((inputs - (1-mask) * loc) * (1-mask) * inverse(scale)) % vocab_size
  ```

  For the reverse pass, the flow computes:

  ```none
  net = layer(mask * inputs)
  loc, scale = tf.split(net, 2, axis=-1)
  loc = tf.argmax(loc, axis=-1)
  scale = tf.argmax(scale, axis=-1)
  outputs = ((1-mask) * loc + (1-mask) * scale * inputs) % vocab_size
  ```

  The modular arithmetic happens in one-hot space.

  If `x` is a discrete random variable, the induced probability mass function on
  the outputs `y = flow(x)` is

  ```none
  p(y) = p(flow.reverse(y)).
  ```

  The location-only transform is always invertible ([integers modulo
  `vocab_size` form an additive group](
  https://en.wikipedia.org/wiki/Modular_arithmetic)). The transform with a scale
  is invertible if the scale and `vocab_size` are coprime (see
  [prime fields](https://en.wikipedia.org/wiki/Finite_field)).
  """

  def __init__(self, layer, mask, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Two-headed masked network taking the inputs and returning a
        real-valued Tensor of shape `[..., length, 2*vocab_size]`.
        Alternatively, `layer` may return a Tensor of shape
        `[..., length, vocab_size]` to be used as the location transform; the
        scale transform will be hard-coded to 1.
      mask: binary Tensor of shape `[length]` forming the bipartite assignment.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(DiscreteBipartiteFlow, self).__init__(**kwargs)
    self.layer = layer
    self.mask = mask
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteBipartiteFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(DiscreteBipartiteFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for bipartite generation."""
    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      inverse_scale = utils.multiplicative_inverse(scale, self.vocab_size)
      shifted_inputs = utils.one_hot_minus(inputs, loc)
      masked_outputs = (1. - mask) * utils.one_hot_multiply(shifted_inputs,
                                                            inverse_scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
      masked_outputs = (1. - mask) * utils.one_hot_minus(inputs, loc)
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    outputs = masked_inputs + masked_outputs
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass for the inverse bipartite transformation."""
    if not self.built:
      self._maybe_build(inputs)

    inputs = tf.convert_to_tensor(inputs)
    batch_ndims = inputs.shape.ndims - 2
    mask = tf.reshape(tf.cast(self.mask, inputs.dtype),
                      [1] * batch_ndims + [-1, 1])
    masked_inputs = mask * inputs
    net = self.layer(masked_inputs, **kwargs)
    if net.shape[-1] == 2 * self.vocab_size:
      loc, scale = tf.split(net, 2, axis=-1)
      scale = tf.cast(utils.one_hot_argmax(scale, self.temperature),
                      inputs.dtype)
      scaled_inputs = utils.one_hot_multiply(inputs, scale)
    elif net.shape[-1] == self.vocab_size:
      loc = net
      scaled_inputs = inputs
    else:
      raise ValueError('Output of layer does not have compatible dimensions.')
    loc = tf.cast(utils.one_hot_argmax(loc, self.temperature), inputs.dtype)
    masked_outputs = (1. - mask) * utils.one_hot_add(loc, scaled_inputs)
    outputs = masked_inputs + masked_outputs
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)


class SinkhornAutoregressiveFlow(tf.python.keras.layers.Layer):
  """A discrete reversible layer using Sinkhorn normalization for permutations.

  The flow takes as input a one-hot Tensor of shape `[..., length, vocab_size]`.
  The flow returns a Tensor of same shape and dtype. (To enable gradients, the
  input must have float dtype.)
  """

  def __init__(self, layer, temperature, **kwargs):
    """Constructs flow.

    Args:
      layer: Masked network taking inputs with shape `[..., length, vocab_size]`
        and returning a real-valued Tensor of shape
        `[..., length, vocab_size ** 2]`. Sinkhorn iterations are applied to
        each `layer` output to produce permutation matrices.
      temperature: Positive value determining bias of gradient estimator.
      **kwargs: kwargs of parent class.
    """
    super(SinkhornAutoregressiveFlow, self).__init__(**kwargs)
    self.layer = layer
    self.temperature = temperature

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.vocab_size = input_shape[-1]
    if self.vocab_size is None:
      raise ValueError('The last dimension of the inputs to '
                       '`DiscreteAutoregressiveFlow` should be defined. Found '
                       '`None`.')
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not isinstance(inputs, random_variable.RandomVariable):
      return super(SinkhornAutoregressiveFlow, self).__call__(
          inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs, **kwargs):
    """Forward pass for left-to-right autoregressive generation."""
    inputs = tf.convert_to_tensor(inputs)
    length = inputs.shape[-2]
    if length is None:
      raise NotImplementedError('length dimension must be known.')
    # Form initial sequence tensor of shape [..., 1, vocab_size]. In a loop, we
    # incrementally build a Tensor of shape [..., t, vocab_size] as t grows.
    outputs = self._initial_call(inputs[..., 0, :], length, **kwargs)
    for t in range(1, length):
      outputs = self._per_timestep_call(outputs,
                                        inputs[..., t, :],
                                        length,
                                        t,
                                        **kwargs)
    return outputs

  def _initial_call(self, new_inputs, length, **kwargs):
    """Returns Tensor of shape [..., 1, vocab_size].

    Args:
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output.
      length: Length of final desired sequence.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = new_inputs[..., tf.newaxis, :]
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs, paddings=[[0, 0]] * batch_ndims + [[0, length - 1], [0, 0]])
    temperature = 1.
    logits = self.layer(padded_inputs / temperature, **kwargs)
    logits = logits[..., 0:1, :]
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = utils.sinkhorn(logits)
    hard = tf.cast(utils.soft_to_hard_permutation(soft), inputs.dtype)
    hard = tf.reshape(hard, logits.shape)
    # Inverse of permutation matrix is its transpose.
    # inputs is [batch_size, timestep + 1, vocab_size].
    # hard is [batch_size, timestep + 1, vocab_size, vocab_size].
    outputs = tf.matmul(inputs[..., tf.newaxis, :],
                        hard,
                        transpose_b=True)[..., 0, :]
    return outputs

  def _per_timestep_call(self,
                         current_outputs,
                         new_inputs,
                         length,
                         timestep,
                         **kwargs):
    """Returns Tensor of shape [..., timestep+1, vocab_size].

    Args:
      current_outputs: Tensor of shape [..., timestep, vocab_size], the so-far
        generated sequence Tensor.
      new_inputs: Tensor of shape [..., vocab_size], the new input to generate
        its output given current_outputs.
      length: Length of final desired sequence.
      timestep: Current timestep.
      **kwargs: Optional keyword arguments to layer.
    """
    inputs = tf.concat([current_outputs,
                        new_inputs[..., tf.newaxis, :]], axis=-2)
    # TODO(trandustin): To handle variable lengths, extend MADE to subset its
    # input and output layer weights rather than pad inputs.
    batch_ndims = inputs.shape.ndims - 2
    padded_inputs = tf.pad(
        inputs,
        paddings=[[0, 0]] * batch_ndims + [[0, length - timestep - 1], [0, 0]])
    logits = self.layer(padded_inputs, **kwargs)
    logits = logits[..., :(timestep+1), :]
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = utils.sinkhorn(logits / self.temperature)
    hard = tf.cast(utils.soft_to_hard_permutation(soft), inputs.dtype)
    hard = tf.reshape(hard, logits.shape)
    # Inverse of permutation matrix is its transpose.
    # inputs is [batch_size, timestep + 1, vocab_size].
    # hard is [batch_size, timestep + 1, vocab_size, vocab_size].
    new_outputs = tf.matmul(inputs[..., tf.newaxis, :],
                            hard,
                            transpose_b=True)[..., 0, :]
    outputs = tf.concat([current_outputs, new_outputs[..., -1:, :]], axis=-2)
    if not tf.executing_eagerly():
      outputs.set_shape([None] * batch_ndims + [timestep+1, self.vocab_size])
    return outputs

  def reverse(self, inputs, **kwargs):
    """Reverse pass returning the inverse autoregressive transformation."""
    if not self.built:
      self._maybe_build(inputs)

    logits = self.layer(inputs, **kwargs)
    logits = tf.reshape(
        logits,
        logits.shape[:-1].concatenate([self.vocab_size, self.vocab_size]))
    soft = utils.sinkhorn(logits / self.temperature, n_iters=20)
    hard = utils.soft_to_hard_permutation(soft)
    hard = tf.reshape(hard, logits.shape)
    # Recover the permutation by right-multiplying by the permutation matrix.
    outputs = tf.matmul(inputs[..., tf.newaxis, :], hard)[..., 0, :]
    return outputs

  def log_det_jacobian(self, inputs):
    return tf.cast(0, inputs.dtype)
