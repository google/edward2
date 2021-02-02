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

"""Layer utilities.

## References:

[1]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
"""

import functools
import random
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

# SciPy is not a mandatory dependency when using the TF backend.
try:
  from scipy.optimize import linear_sum_assignment  # pylint: disable=g-import-not-at-top
except ImportError:
  pass


def add_weight(cls):
  """Decorator for Layers, overriding add_weight for trainable initializers."""
  @functools.wraps(cls.add_weight)
  def _add_weight(self,
                  name=None,
                  shape=None,
                  dtype=None,
                  initializer=None,
                  regularizer=None,
                  **kwargs):
    """Adds weight."""
    # Rely on the keras trackable machinery to pick up weights where applicable.
    # The name for the field is arbitrary.
    if getattr(self, 'tracked_add_weight_dependencies', None) is None:
      self.tracked_add_weight_dependencies = []
    self.tracked_add_weight_dependencies.append((regularizer, initializer))

    if isinstance(regularizer, tf.python.keras.layers.Layer):
      if not regularizer.built:
        regularizer.build(shape)
    if isinstance(initializer, tf.python.keras.layers.Layer):
      with tf.name_scope(name):
        weight = initializer(shape, dtype)
      if regularizer is not None:
        def loss_fn():
          """Creates a regularization loss `Tensor`."""
          with tf.name_scope(name + '/Regularizer'):
            return regularizer(initializer(shape, dtype))
        self.add_loss(loss_fn)
      return weight
    return super(cls, self).add_weight(name=name,
                                       shape=shape,
                                       dtype=dtype,
                                       initializer=initializer,
                                       regularizer=regularizer,
                                       **kwargs)
  cls.add_weight = _add_weight
  return cls


def one_hot_argmax(inputs, temperature, axis=-1):
  """Returns one-hot of argmax with backward pass set to softmax-temperature."""
  vocab_size = inputs.shape[-1]
  hard = tf.one_hot(tf.argmax(inputs, axis=axis),
                    depth=vocab_size,
                    axis=axis,
                    dtype=inputs.dtype)
  soft = tf.nn.softmax(inputs / temperature, axis=axis)
  outputs = soft + tf.stop_gradient(hard - soft)
  return outputs


def one_hot_add(inputs, shift):
  """Performs (inputs + shift) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to shift the corresponding one-hot vector in
      inputs. Soft values perform a "weighted shift": for example,
      shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
      zero; 0.3 * shifting by one; and 0.5 * shifting by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # Compute circular 1-D convolution with shift as the kernel.
  inputs = tf.cast(inputs, tf.complex64)
  shift = tf.cast(shift, tf.complex64)
  return tf.math.real(
      tf.signal.ifft(tf.signal.fft(inputs) * tf.signal.fft(shift)))


def one_hot_minus(inputs, shift):
  """Performs (inputs - shift) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    shift: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to shift the corresponding one-hot vector in
      inputs. Soft values perform a "weighted shift": for example,
      shift=[0.2, 0.3, 0.5] performs a linear combination of 0.2 * shifting by
      zero; 0.3 * shifting by one; and 0.5 * shifting by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # TODO(trandustin): Implement with circular conv1d.
  inputs = tf.convert_to_tensor(inputs)
  shift = tf.cast(shift, inputs.dtype)
  vocab_size = inputs.shape[-1]
  # Form a [..., vocab_size, vocab_size] matrix. Each batch element of
  # inputs will vector-matrix multiply the vocab_size x vocab_size matrix. This
  # "shifts" the inputs batch element by the corresponding shift batch element.
  shift_matrix = tf.stack([tf.roll(shift, i, axis=-1)
                           for i in range(vocab_size)], axis=-2)
  outputs = tf.einsum('...v,...uv->...u', inputs, shift_matrix)
  return outputs


def one_hot_multiply(inputs, scale):
  """Performs (inputs * scale) % vocab_size in the one-hot space.

  Args:
    inputs: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor.
    scale: Tensor of shape `[..., vocab_size]`. Typically a soft/hard one-hot
      Tensor specifying how much to scale the corresponding one-hot vector in
      inputs. Soft values perform a "weighted scale": for example,
      scale=[0.2, 0.3, 0.5] performs a linear combination of
      0.2 * scaling by zero; 0.3 * scaling by one; and 0.5 * scaling by two.

  Returns:
    Tensor of same shape and dtype as inputs.
  """
  # TODO(trandustin): Implement with circular conv1d.
  inputs = tf.convert_to_tensor(inputs)
  scale = tf.cast(scale, inputs.dtype)
  batch_shape = inputs.shape[:-1].as_list()
  vocab_size = inputs.shape[-1]
  # Form a [..., vocab_size, vocab_size] tensor. The ith row of the
  # batched vocab_size x vocab_size matrix represents scaling inputs by i.
  permutation_matrix = tf.math.floormod(
      tf.tile(tf.range(vocab_size)[:, tf.newaxis], [1, vocab_size]) *
      tf.range(vocab_size)[tf.newaxis], vocab_size)
  permutation_matrix = tf.cast(
      tf.one_hot(permutation_matrix, depth=vocab_size, axis=-1), inputs.dtype)
  # Scale the inputs according to the permutation matrix of all possible scales.
  scaled_inputs = tf.einsum('...v,avu->...au', inputs, permutation_matrix)
  scaled_inputs = tf.concat([
      tf.zeros(batch_shape + [1, vocab_size], dtype=inputs.dtype),
      scaled_inputs[..., 1:, :]], axis=-2)
  # Reduce rows of the scaled inputs by the scale values. This forms a
  # weighted linear combination of scaling by zero, scaling by one, and so on.
  outputs = tf.einsum('...v,...vu->...u', scale, scaled_inputs)
  return outputs


def py_multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n (in Python).

  Implements extended Euclidean algorithm.

  Args:
    a: int-like np.ndarray.
    n: int.

  Returns:
    Multiplicative inverse as an int32 np.ndarray with same shape as a.
  """
  batched_a = np.asarray(a, dtype=np.int32)
  batched_inverse = []
  for a in np.nditer(batched_a):
    inverse = 0
    new_inverse = 1
    remainder = n
    new_remainder = a
    while new_remainder != 0:
      quotient = remainder // new_remainder
      (inverse, new_inverse) = (new_inverse, inverse - quotient * new_inverse)
      (remainder, new_remainder) = (new_remainder,
                                    remainder - quotient * new_remainder)
    if remainder > 1:
      return ValueError(
          'Inverse for {} modulo {} does not exist.'.format(a, n))
    if inverse < 0:
      inverse += n
    batched_inverse.append(inverse)
  return np.asarray(batched_inverse, dtype=np.int32).reshape(batched_a.shape)


def multiplicative_inverse(a, n):
  """Multiplicative inverse of a modulo n.

  Args:
    a: Tensor of shape [..., vocab_size]. It denotes an integer in the one-hot
      space.
    n: int Tensor of shape [...].

  Returns:
    Tensor of same shape and dtype as a.
  """
  a = tf.convert_to_tensor(a)
  n = tf.convert_to_tensor(n)
  vocab_size = a.shape[-1]
  a_dtype = a.dtype
  sparse_a = tf.argmax(a, axis=-1)
  # TODO(trandustin): Change to tf.py_function.
  sparse_outputs = tf1.py_func(
      py_multiplicative_inverse, [sparse_a, n], tf.int32)
  sparse_outputs.set_shape(sparse_a.shape)
  outputs = tf.one_hot(sparse_outputs, depth=vocab_size, dtype=a_dtype)
  return outputs


def soft_to_hard_permutation(inputs):
  """Returns permutation matrices by solving a matching problem.

  Solves linear sum assignment to convert doubly-stochastic matrices to
  permutation matrices. It uses scipy.optimize.linear_sum_assignment to solve
  the optimization problem max_P sum_i,j M_i,j P_i,j with P a permutation
  matrix. Notice the negative sign; the reason, the original function solves a
  minimization problem.

  Code is adapted from Mena et al. [1].

  [1] Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek.
  Learning latent permutations with Gumbel-Sinkhorn networks. International
  Conference on Learning Representations, 2018.

  Args:
    inputs: A `Tensor` with shape `[:, vocab_size, vocab_size]` that is
      doubly-stochastic in its last two dimensions.

  Returns:
    outputs: A hard permutation `Tensor` with the same shape as `inputs` (in
      other words the last two dimensions are doubly-stochastic and each element
      is 0 or 1).
  """

  def hungarian(x):
    """Hungarian algorithm."""
    x = x.numpy()
    if x.ndim == 2:
      x = np.reshape(x, [1, x.shape[0], x.shape[1]])
    sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
    for i in range(x.shape[0]):
      try:
        sol[i, :] = linear_sum_assignment(-x[i, :])[1].astype(np.int32)
      except NameError:
        raise NameError('linear_sum_assignment requires SciPy to be installed.')
    return tf.convert_to_tensor(sol)

  vocab_size = inputs.shape[-1]
  # Note: tf.py_function isn't currently supported on headless GPUs.
  # TODO(vafa): Fix tf.py_function headless GPU bug.
  permutation_lists = tf.py_function(hungarian, [inputs], tf.int32)
  hard = tf.one_hot(permutation_lists, depth=vocab_size)
  outputs = tf.stop_gradient(hard - inputs) + inputs
  return outputs


def sinkhorn(inputs, n_iters=20):
  """Performs incomplete Sinkhorn normalization to inputs.

  By a theorem by Sinkhorn and Knopp [1], a sufficiently well-behaved  matrix
  with positive entries can be turned into a doubly-stochastic matrix
  (i.e. its rows and columns add up to one) via the succesive row and column
  normalization.
  -To ensure positivity, the effective input to sinkhorn has to be
  exp(inputs) (elementwise).
  -However, for stability, sinkhorn works in the log-space. It is only at
   return time that entries are exponentiated.

  Code is adapted from Mena et al. [2].

  [1] Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and
  doubly stochastic matrices. Pacific Journal of Mathematics, 1967.

  [2] Gonzalo Mena, David Belanger, Scott Linderman, Jasper Snoek.
  Learning latent permutations with Gumbel-Sinkhorn networks. International
  Conference on Learning Representations, 2018.

  Args:
    inputs: A `Tensor` with shape `[..., vocab_size, vocab_size]`.
    n_iters: Number of sinkhorn iterations (in practice, as little as 20
      iterations are needed to achieve decent convergence for `vocab_size` ~100)

  Returns:
    outputs: A `Tensor` of close-to-doubly-stochastic matrices with shape
      `[:, vocab_size, vocab_size]`.
  """
  vocab_size = tf.shape(inputs)[-1]
  log_alpha = tf.reshape(inputs, [-1, vocab_size, vocab_size])

  for _ in range(n_iters):
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=2),
                            [-1, vocab_size, 1])
    log_alpha -= tf.reshape(tf.reduce_logsumexp(log_alpha, axis=1),
                            [-1, 1, vocab_size])
  outputs = tf.exp(log_alpha)
  return outputs


# From `tensorflow/python/framework/smart_cond.py`
def smart_constant_value(pred):
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Arguments:
    pred: A scalar, either a Python bool or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Tensor or bool.
  """
  if pred in {0, 1}:  # Accept 1/0 as valid boolean values
    pred_value = bool(pred)
  elif isinstance(pred, bool):
    pred_value = pred
  elif tf.is_tensor(pred):
    pred_value = tf.get_static_value(pred)
  else:
    raise TypeError('`pred` must be a Tensor, or a Python bool, or 1 or 0. '
                    'Found instead: %s' % pred)
  return pred_value


def mean_field_logits(logits,
                      covmat=None,
                      mean_field_factor=1.,
                      likelihood='logistic'):
  """Adjust the model logits so its softmax approximates the posterior mean [1].

  Arguments:
    logits: A float tensor of shape (batch_size, num_classes).
    covmat: A float tensor of shape (batch_size, batch_size). If None then it
      assumes the covmat is an identity matrix.
    mean_field_factor: The scale factor for mean-field approximation, used to
      adjust the influence of posterior variance in posterior mean
      approximation. If covmat=None then it is used as the scaling parameter for
      temperature scaling.
    likelihood: Likelihood for integration in Gaussian-approximated latent
      posterior.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  """
  if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
    raise ValueError(
        f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
    )

  if mean_field_factor < 0:
    return logits

  # Compute standard deviation.
  if covmat is None:
    variances = 1.
  else:
    variances = tf.linalg.diag_part(covmat)

  # Compute scaling coefficient for mean-field approximation.
  if likelihood == 'poisson':
    logits_scale = tf.exp(- variances * mean_field_factor / 2.)
  else:
    logits_scale = tf.sqrt(1. + variances * mean_field_factor)

  # Cast logits_scale to compatible dimension.
  if len(logits.shape) > 1:
    logits_scale = tf.expand_dims(logits_scale, axis=-1)

  return logits / logits_scale


def gen_int_seed():
  return random.randrange(2**63 - 1)
