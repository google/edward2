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

# Lint as: python3
"""Bayesian recurrent cells and layers."""

from typing import List, Tuple, Union  # pylint:disable=g-bad-import-order

from edward2.tensorflow import constraints
from edward2.tensorflow import initializers
from edward2.tensorflow import random_variable
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow as tf


@utils.add_weight
class LSTMCellReparameterization(tf.python.keras.layers.LSTMCell):
  """Bayesian LSTM cell class estimated via reparameterization.

  The layer computes a variational Bayesian approximation to the distribution
  over LSTM cell functions,

  ```
  p(outputs | inputs) = int lstm_cell(inputs; weights, bias) p(weights, bias)
    dweights dbias,
  ```

  where the weights consist of both input and recurrent weights.

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel, recurrent kernel, and bias. Gradients with
  respect to the distributions' learnable parameters backpropagate via
  reparameterization.  Minimizing cross-entropy plus the layer's losses performs
  variational minimum description length, i.e., it minimizes an upper bound to
  the negative marginal likelihood.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='trainable_normal',
               recurrent_initializer='trainable_normal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer='normal_kl_divergence',
               recurrent_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               **kwargs):
    self.called_weights = False
    super(LSTMCellReparameterization, self).__init__(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        recurrent_initializer=initializers.get(recurrent_initializer),
        bias_initializer=initializers.get(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=regularizers.get(kernel_regularizer),
        recurrent_regularizer=regularizers.get(recurrent_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        recurrent_constraint=constraints.get(recurrent_constraint),
        bias_constraint=constraints.get(bias_constraint),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        **kwargs)

  # TODO(dusenberrymw): TensorFlow has an open RFC
  # (https://github.com/tensorflow/community/pull/208) for adding core TF types
  # to the library such that the library and end users can upgrade to the use of
  # type annotations. Once that RFC is accepted and implemented, we should
  # update these types below.
  def build(
      self, input_shape: Union[tf.TensorShape, List[int], Tuple[int, ...]]
  ) -> None:
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if (self.unit_forget_bias and not isinstance(self.bias_initializer,
                                                   tf.python.keras.layers.Layer)):
        def bias_initializer(_, *args, **kwargs):
          return tf.python.keras.backend.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              tf.python.keras.initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def call(self, *args, **kwargs):
    if not self.called_weights:
      # Call weights if never called before. This ensures TF ops executed during
      # the cell's first ever call (e.g., constraints applied to free parameters
      # in the variational distribution) are recorded properly on any gradient
      # tape. Unlike variational dense or convolutional layers, LSTM cell weight
      # noise is reused across calls (i.e., timesteps). Call get_initial_state()
      # or call_weights() explicitly to get a new sample of the weights.
      self.call_weights()
    return super(LSTMCellReparameterization, self).call(*args, **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.python.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.recurrent_initializer, tf.python.keras.layers.Layer):
      self.recurrent_kernel = self.recurrent_initializer(
          self.recurrent_kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.python.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)
    self.called_weights = True

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self.call_weights()
    return super(LSTMCellReparameterization, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    # Index the associated tensor as indexing does not work over the event shape
    # of distributions.
    recurrent_kernel = tf.convert_to_tensor(self.recurrent_kernel)
    i = self.recurrent_activation(
        x_i + tf.python.keras.backend.dot(h_tm1_i, recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + tf.python.keras.backend.dot(
        h_tm1_f, recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + tf.python.keras.backend.dot(
        h_tm1_c, recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + tf.python.keras.backend.dot(
            h_tm1_o, recurrent_kernel[:, self.units * 3:]))
    return c, o


class LSTMCellFlipout(LSTMCellReparameterization):
  """Bayesian LSTM cell class estimated via Flipout (Wen et al., 2018).

  The layer computes a variational Bayesian approximation to the distribution
  over LSTM cell functions,

  ```
  p(outputs | inputs) = int lstm_cell(inputs; weights, bias) p(weights, bias)
    dweights dbias,
  ```

  where the weights consist of both input and recurrent weights.

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel, recurrent kernel, and bias. Gradients with
  respect to the distributions' learnable parameters backpropagate via
  reparameterization.  Minimizing cross-entropy plus the layer's losses performs
  variational minimum description length, i.e., it minimizes an upper bound to
  the negative marginal likelihood.

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel` and `recurrent_kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` and `recurrent_kernel` random
  variable must be able to decompose as a sum of its mean and a perturbation
  distribution; the perturbation distribution must be independent across weight
  elements and symmetric around zero (for example, a fully factorized Gaussian).
  """

  def _call_sign_flips(self, inputs=None, batch_size=None, dtype=None):
    """Builds per-example sign flips for pseudo-independent perturbations."""
    # We add and call this method separately from build(). build() operates on a
    # static input_shape, and we need the batch size which is often dynamic.
    if inputs is not None:
      batch_size = tf.shape(inputs)[0]
      dtype = inputs.dtype
    input_dim = self.kernel.shape[0]
    self.sign_input = 2 * tf.random.uniform(
        [batch_size, input_dim], minval=0, maxval=2, dtype=tf.int32) - 1
    self.sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=tf.int32) - 1
    self.recurrent_sign_input = 2 * tf.random.uniform(
        [batch_size, self.units], minval=0, maxval=2, dtype=tf.int32) - 1
    self.recurrent_sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=tf.int32) - 1
    self.sign_input = tf.cast(self.sign_input, dtype)
    self.sign_output = tf.cast(self.sign_output, dtype)
    self.recurrent_sign_input = tf.cast(self.recurrent_sign_input, dtype)
    self.recurrent_sign_output = tf.cast(self.recurrent_sign_output, dtype)

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self._call_sign_flips(inputs, batch_size, dtype)
    return super(LSTMCellFlipout, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    if not isinstance(self.recurrent_kernel, random_variable.RandomVariable):
      return super(LSTMCellFlipout, self)._compute_carry_and_output(x,
                                                                    h_tm1,
                                                                    c_tm1)
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    kernel_mean = self.recurrent_kernel.distribution.mean()
    perturbation = self.recurrent_kernel - kernel_mean
    k_i, k_f, k_c, k_o = tf.split(kernel_mean, num_or_size_splits=4, axis=1)
    p_i, p_f, p_c, p_o = tf.split(perturbation, num_or_size_splits=4, axis=1)
    so_i, so_f, so_c, so_o = tf.split(self.recurrent_sign_output,
                                      num_or_size_splits=4, axis=1)
    z0 = (x_i + tf.python.keras.backend.dot(h_tm1_i, k_i) +
          tf.python.keras.backend.dot(h_tm1_i * self.recurrent_sign_input, p_i) * so_i)
    z1 = (x_f + tf.python.keras.backend.dot(h_tm1_f, k_f) +
          tf.python.keras.backend.dot(h_tm1_f * self.recurrent_sign_input, p_f) * so_f)
    z2 = (x_c + tf.python.keras.backend.dot(h_tm1_c, k_c) +
          tf.python.keras.backend.dot(h_tm1_c * self.recurrent_sign_input, p_c) * so_c)
    z3 = (x_o + tf.python.keras.backend.dot(h_tm1_o, k_o) +
          tf.python.keras.backend.dot(h_tm1_o * self.recurrent_sign_input, p_o) * so_o)
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    # TODO(trandustin): Enable option for Flipout on only the kernel or
    # recurrent_kernel. If only one is a random variable, we currently default
    # to weight reparameterization.
    if (not isinstance(self.kernel, random_variable.RandomVariable) or
        not isinstance(self.recurrent_kernel, random_variable.RandomVariable)):
      return super(LSTMCellFlipout, self).call(inputs, states, training)
    if not self.called_weights:
      self.call_weights()
      self._call_sign_flips(inputs)
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      kernel_mean = self.kernel.distribution.mean()
      perturbation = self.kernel - kernel_mean
      k_i, k_f, k_c, k_o = tf.split(kernel_mean, num_or_size_splits=4, axis=1)
      p_i, p_f, p_c, p_o = tf.split(perturbation, num_or_size_splits=4, axis=1)
      so_i, so_f, so_c, so_o = tf.split(self.sign_output,
                                        num_or_size_splits=4, axis=1)
      x_i = (tf.python.keras.backend.dot(inputs_i, k_i) +
             tf.python.keras.backend.dot(inputs_i * self.sign_input, p_i) * so_i)
      x_f = (tf.python.keras.backend.dot(inputs_f, k_f) +
             tf.python.keras.backend.dot(inputs_f * self.sign_input, p_f) * so_f)
      x_c = (tf.python.keras.backend.dot(inputs_c, k_c) +
             tf.python.keras.backend.dot(inputs_c * self.sign_input, p_c) * so_c)
      x_o = (tf.python.keras.backend.dot(inputs_o, k_o) +
             tf.python.keras.backend.dot(inputs_o * self.sign_input, p_o) * so_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = tf.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = tf.python.keras.backend.bias_add(x_i, b_i)
        x_f = tf.python.keras.backend.bias_add(x_f, b_f)
        x_c = tf.python.keras.backend.bias_add(x_c, b_c)
        x_o = tf.python.keras.backend.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      kernel_mean = self.kernel.distribution.mean()
      perturbation = self.kernel - kernel_mean
      z = tf.python.keras.backend.dot(inputs, kernel_mean)
      z += tf.python.keras.backend.dot(inputs * self.sign_input,
                                perturbation) * self.sign_output
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 = h_tm1 * rec_dp_mask[0]
      recurrent_kernel_mean = self.recurrent_kernel.distribution.mean()
      perturbation = self.recurrent_kernel - recurrent_kernel_mean
      z += tf.python.keras.backend.dot(h_tm1, recurrent_kernel_mean)
      z += tf.python.keras.backend.dot(h_tm1 * self.recurrent_sign_input,
                                perturbation) * self.recurrent_sign_output
      if self.use_bias:
        z = tf.python.keras.backend.bias_add(z, self.bias)

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]


@utils.add_weight
class LSTMCellRank1(tf.python.keras.layers.LSTMCell):
  """A rank-1 Bayesian neural net LSTM cell layer (Dusenberry et al., 2020).

  The layer computes a variational Bayesian approximation to the distribution
  over LSTM cell functions,

  ```
  p(outputs | inputs) = int lstm_cell(inputs; weights, bias) p(weights, bias)
    dweights dbias,
  ```

  where the weights consist of both input and recurrent weights.

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel, recurrent kernel, and bias. Gradients with
  respect to the distributions' learnable parameters backpropagate via
  reparameterization.  Minimizing cross-entropy plus the layer's losses performs
  variational minimum description length, i.e., it minimizes an upper bound to
  the negative marginal likelihood.

  This layer uses the rank-1 setup whereby the priors and variational
  distributions are over a rank-1 subspace of weights, where each weight matrix
  is decomposed into the product of a matrix and the outer product of two
  vectors, alpha and gamma. Rank-1 priors posit distributions over the alpha and
  gamma vectors.

  The argument ensemble_size selects the number of mixture components over all
  weights, i.e., an ensemble of size `ensemble_size`. The layer performs a
  forward pass by enumeration, returning a forward pass under each mixture
  component. It takes an input tensor of shape
  [ensemble_size*examples_per_model, input_dim] and returns an output tensor of
  shape [ensemble_size*examples_per_model, units].

  To use a different batch for each mixture, take a minibatch of size
  ensemble_size*examples_per_model. To use the same batch for each mixture, get
  a minibatch of size examples_per_model and tile it by ensemble_size before
  applying any ensemble layers.
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               alpha_initializer='trainable_normal',
               gamma_initializer='trainable_normal',
               kernel_initializer='glorot_uniform',
               recurrent_alpha_initializer='trainable_normal',
               recurrent_gamma_initializer='trainable_normal',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               alpha_regularizer='normal_kl_divergence',
               gamma_regularizer='normal_kl_divergence',
               kernel_regularizer=None,
               recurrent_alpha_regularizer='normal_kl_divergence',
               recurrent_gamma_regularizer='normal_kl_divergence',
               recurrent_regularizer=None,
               bias_regularizer=None,
               alpha_constraint=None,
               gamma_constraint=None,
               kernel_constraint=None,
               recurrent_alpha_constraint=None,
               recurrent_gamma_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               use_additive_perturbation=False,
               ensemble_size=1,
               **kwargs):
    """Initializes an LSTM cell layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use. If you pass `None`, no activation
        is applied (ie. "linear" activation is `a(x) = x`).
      recurrent_activation: Activation function to use for the recurrent step.
         If you pass `None`, no activation is applied (ie. "linear" activation
         is `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      alpha_initializer: Initializer for the rank-1 weights vector applied to
        the inputs before the `kernel` is applied.
      gamma_initializer: Initializer for the rank-1 weights vector applied to
        after the `kernel` is applied.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_alpha_initializer: Initializer for the rank-1 weights vector
        applied to the recurrent state before the `recurrent_kernel` is applied.
      recurrent_gamma_initializer: Initializer for the rank-1 weights vector
        applied after the `recurrent_kernel` is applied.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      unit_forget_bias: Boolean (default `True`). If True, add 1 to the bias of
        the forget gate at initialization. Setting it to true will also force
        `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
          al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      alpha_regularizer: Regularizer function applied to the `alpha` weights
        vector.
      gamma_regularizer: Regularizer function applied to the `gamma` weights
        vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_alpha_regularizer: Regularizer function applied to the
        `recurrent_alpha` weights vector.
      recurrent_gamma_regularizer: Regularizer function applied to the
        `recurrent_gamma` weights vector.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      alpha_constraint: Constraint function applied to the `alpha` weights
        vector.
      gamma_constraint: Constraint function applied to the `gamma` weights
        vector.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_alpha_constraint: Constraint function applied to the
        `recurrent_alpha` weights vector.
      recurrent_gamma_constraint: Constraint function applied to the
        `recurrent_gamma` weights vector.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of smaller dot
        products and additions, whereas mode 2 (default) will batch them into
        fewer, larger operations. These modes will have different performance
        profiles on different hardware and for different applications.
      use_additive_perturbation: Whether or not to use additive interactions vs.
        multiplicative actions.
      ensemble_size: Number of ensemble members, or equivalently, number of
        mixture components.
      **kwargs: Any additional arguments to pass to the superclasses.
    """
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.recurrent_alpha_initializer = initializers.get(
        recurrent_alpha_initializer)
    self.recurrent_gamma_initializer = initializers.get(
        recurrent_gamma_initializer)
    self.alpha_regularizer = regularizers.get(alpha_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.recurrent_alpha_regularizer = regularizers.get(
        recurrent_alpha_regularizer)
    self.recurrent_gamma_regularizer = regularizers.get(
        recurrent_gamma_regularizer)
    self.alpha_constraint = constraints.get(alpha_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.recurrent_alpha_constraint = constraints.get(
        recurrent_alpha_constraint)
    self.recurrent_gamma_constraint = constraints.get(
        recurrent_gamma_constraint)
    self.use_additive_perturbation = use_additive_perturbation
    self.ensemble_size = ensemble_size
    self.sampled_weights = False
    super().__init__(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=initializers.get(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=regularizers.get(bias_regularizer),
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=constraints.get(bias_constraint),
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        **kwargs)

  def build(self, input_shape):
    """Creates the variables of the layer."""
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    self.alpha = self.add_weight(
        name='alpha',
        shape=[self.ensemble_size, input_dim],
        initializer=self.alpha_initializer,
        regularizer=self.alpha_regularizer,
        constraint=self.alpha_constraint,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        name='gamma',
        shape=[self.ensemble_size, self.units * 4],
        initializer=self.gamma_initializer,
        regularizer=self.gamma_regularizer,
        constraint=self.gamma_constraint,
        trainable=True,
        dtype=self.dtype)
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_alpha = self.add_weight(
        name='recurrent_alpha',
        shape=[self.ensemble_size, self.units],
        initializer=self.recurrent_alpha_initializer,
        regularizer=self.recurrent_alpha_regularizer,
        constraint=self.recurrent_alpha_constraint,
        trainable=True,
        dtype=self.dtype)
    self.recurrent_gamma = self.add_weight(
        name='recurrent_gamma',
        shape=[self.ensemble_size, self.units * 4],
        initializer=self.recurrent_gamma_initializer,
        regularizer=self.recurrent_gamma_regularizer,
        constraint=self.recurrent_gamma_constraint,
        trainable=True,
        dtype=self.dtype)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if (self.unit_forget_bias and not isinstance(self.bias_initializer,
                                                   tf.python.keras.layers.Layer)):
        def bias_initializer(_, *args, **kwargs):
          return tf.concat([
              self.bias_initializer([self.ensemble_size, self.units], *args,
                                    **kwargs),
              tf.python.keras.initializers.Ones()([self.ensemble_size, self.units],
                                           *args, **kwargs),
              self.bias_initializer([self.ensemble_size, self.units * 2],
                                    *args, **kwargs),
          ], axis=1)
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=[self.ensemble_size, self.units * 4],
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
      self.bias_shape = self.bias.shape
    self.alpha_shape = self.alpha.shape
    self.gamma_shape = self.gamma.shape
    self.recurrent_alpha_shape = self.recurrent_alpha.shape
    self.recurrent_gamma_shape = self.recurrent_gamma.shape

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    rec_k_i, rec_k_f, rec_k_c, rec_k_o = tf.split(
        self.recurrent_kernel, num_or_size_splits=4, axis=1)
    rec_alpha = self.recurrent_alpha_sample
    rec_gamma_i, rec_gamma_f, rec_gamma_c, rec_gamma_o = tf.split(
        self.recurrent_gamma_sample, num_or_size_splits=4, axis=1)
    if self.use_additive_perturbation:
      rec_i = tf.linalg.matmul(h_tm1_i + rec_alpha, rec_k_i) + rec_gamma_i
      rec_f = tf.linalg.matmul(h_tm1_f + rec_alpha, rec_k_f) + rec_gamma_f
      rec_c = tf.linalg.matmul(h_tm1_c + rec_alpha, rec_k_c) + rec_gamma_c
      rec_o = tf.linalg.matmul(h_tm1_o + rec_alpha, rec_k_o) + rec_gamma_o
    else:
      rec_i = tf.linalg.matmul(h_tm1_i * rec_alpha, rec_k_i) * rec_gamma_i
      rec_f = tf.linalg.matmul(h_tm1_f * rec_alpha, rec_k_f) * rec_gamma_f
      rec_c = tf.linalg.matmul(h_tm1_c * rec_alpha, rec_k_c) * rec_gamma_c
      rec_o = tf.linalg.matmul(h_tm1_o * rec_alpha, rec_k_o) * rec_gamma_o
    i = self.recurrent_activation(x_i + rec_i)
    f = self.recurrent_activation(x_f + rec_f)
    c = f * c_tm1 + i * self.activation(x_c + rec_c)
    o = self.recurrent_activation(x_o + rec_o)
    return c, o

  def _sample_weights(self, inputs=None, batch_size=None, dtype=None):
    """Samples any rank-1 weight tensor if the initializer is itself a layer."""
    if inputs is not None:
      batch_size = tf.shape(inputs)[0]
    examples_per_model = batch_size // self.ensemble_size

    # Sample parameters for each input example.
    def sample(weight_variable, weight_initializer, shape):
      if isinstance(weight_initializer, tf.python.keras.layers.Layer):
        weights = weight_initializer(
            shape, self.dtype).distribution.sample(examples_per_model)
        weights = tf.transpose(weights, [1, 0, 2])
      else:
        weights = tf.tile(weight_variable, [1, examples_per_model])
      return weights

    alpha = sample(self.alpha, self.alpha_initializer, self.alpha_shape)
    gamma = sample(self.gamma, self.gamma_initializer, self.gamma_shape)
    recurrent_alpha = sample(self.recurrent_alpha,
                             self.recurrent_alpha_initializer,
                             self.recurrent_alpha_shape)
    recurrent_gamma = sample(self.recurrent_gamma,
                             self.recurrent_gamma_initializer,
                             self.recurrent_gamma_shape)

    self.alpha_sample = tf.reshape(alpha, [batch_size, -1])
    self.gamma_sample = tf.reshape(gamma, [batch_size, -1])
    self.recurrent_alpha_sample = tf.reshape(recurrent_alpha, [batch_size, -1])
    self.recurrent_gamma_sample = tf.reshape(recurrent_gamma, [batch_size, -1])
    if self.use_bias:
      bias = sample(self.bias, self.bias_initializer, self.bias_shape)
      self.bias_sample = tf.reshape(bias, [batch_size, -1])
    self.sampled_weights = True

  def call(self, inputs, states, training=None):
    """Executes the forward pass of the layer."""
    batch_size = tf.shape(inputs)[0]
    if not self.sampled_weights:
      self._sample_weights(batch_size=batch_size)

    alpha = self.alpha_sample
    gamma = self.gamma_sample
    recurrent_alpha = self.recurrent_alpha_sample
    recurrent_gamma = self.recurrent_gamma_sample
    if self.use_bias:
      bias = self.bias_sample

    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
      gamma_i, gamma_f, gamma_c, gamma_o = tf.split(
          gamma, num_or_size_splits=4, axis=1)
      if self.use_additive_perturbation:
        x_i = tf.linalg.matmul(inputs_i + alpha, k_i) + gamma_i
        x_f = tf.linalg.matmul(inputs_f + alpha, k_f) + gamma_f
        x_c = tf.linalg.matmul(inputs_c + alpha, k_c) + gamma_c
        x_o = tf.linalg.matmul(inputs_o + alpha, k_o) + gamma_o
      else:
        x_i = tf.linalg.matmul(inputs_i * alpha, k_i) * gamma_i
        x_f = tf.linalg.matmul(inputs_f * alpha, k_f) * gamma_f
        x_c = tf.linalg.matmul(inputs_c * alpha, k_c) * gamma_c
        x_o = tf.linalg.matmul(inputs_o * alpha, k_o) * gamma_o
      if self.use_bias:
        b_i, b_f, b_c, b_o = tf.split(bias, num_or_size_splits=4, axis=1)
        x_i += b_i
        x_f += b_f
        x_c += b_c
        x_o += b_o

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]
      if self.use_additive_perturbation:
        z = tf.linalg.matmul(inputs + alpha, self.kernel) + gamma
        z += tf.linalg.matmul(
            h_tm1 + recurrent_alpha, self.recurrent_kernel) + recurrent_gamma
      else:
        z = tf.linalg.matmul(inputs * alpha, self.kernel) * gamma
        z += tf.linalg.matmul(
            h_tm1 * recurrent_alpha, self.recurrent_kernel) * recurrent_gamma
      if self.use_bias:
        z += bias

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]

  def get_config(self):
    """Returns the configuration for the layer."""
    config = {
        'units': self.units,
        'activation': tf.python.keras.activations.serialize(self.activation),
        'recurrent_activation': tf.python.keras.activations.serialize(
            self.recurrent_activation),
        'use_bias': self.use_bias,
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'kernel_initializer': initializers.serialize(
            self.kernel_initializer),
        'recurrent_alpha_initializer': initializers.serialize(
            self.recurrent_alpha_initializer),
        'recurrent_gamma_initializer': initializers.serialize(
            self.recurrent_gamma_initializer),
        'recurrent_initializer': initializers.serialize(
            self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'unit_forget_bias': self.unit_forget_bias,
        'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
        'kernel_regularizer': regularizers.serialize(
            self.kernel_regularizer),
        'recurrent_alpha_regularizer': regularizers.serialize(
            self.recurrent_alpha_regularizer),
        'recurrent_gamma_regularizer': regularizers.serialize(
            self.recurrent_gamma_regularizer),
        'recurrent_regularizer': regularizers.serialize(
            self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'alpha_constraint': constraints.serialize(self.alpha_constraint),
        'gamma_constraint': constraints.serialize(self.gamma_constraint),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_alpha_constraint': constraints.serialize(
            self.recurrent_alpha_constraint),
        'recurrent_gamma_constraint': constraints.serialize(
            self.recurrent_gamma_constraint),
        'recurrent_constraint': constraints.serialize(
            self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout,
        'implementation': self.implementation,
        'use_additive_perturbation': self.use_additive_perturbation,
        'ensemble_size': self.ensemble_size,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Gets the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self._sample_weights(inputs, batch_size, dtype)
    return super().get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)

  def set_weights(self, *args, **kwargs):
    """Sets the weights of the layer, from Numpy arrays."""
    self.sampled_weights = False
    super().set_weights(*args, **kwargs)
