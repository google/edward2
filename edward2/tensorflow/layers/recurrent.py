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

"""Bayesian recurrent cells and layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow import constraints
from edward2.tensorflow import initializers
from edward2.tensorflow import random_variable
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


@utils.add_weight
class LSTMCellReparameterization(tf.keras.layers.LSTMCell):
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
               recurrent_activation='hard_sigmoid',
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
               implementation=1,
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

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    if isinstance(input_dim, tf1.Dimension):
      input_dim = input_dim.value
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
      if self.unit_forget_bias:
        if isinstance(self.bias_initializer, tf.keras.layers.Layer):
          def bias_mean_initializer(_, *args, **kwargs):
            return tf.concat([
                tf.keras.initializers.TruncatedNormal(
                    stddev=1e-5)((self.units,), *args, **kwargs),
                tf.keras.initializers.TruncatedNormal(
                    mean=1., stddev=1e-5)((self.units,), *args, **kwargs),
                tf.keras.initializers.TruncatedNormal(
                    stddev=1e-5)((self.units * 2,), *args, **kwargs),
            ], axis=0)
          bias_initializer = initializers.TrainableNormal(
              mean_initializer=bias_mean_initializer)
        else:
          def bias_initializer(_, *args, **kwargs):
            return tf.keras.backend.concatenate([
                self.bias_initializer((self.units,), *args, **kwargs),
                tf.keras.initializers.Ones()((self.units,), *args, **kwargs),
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
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.recurrent_initializer, tf.keras.layers.Layer):
      self.recurrent_kernel = self.recurrent_initializer(
          self.recurrent_kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)
    self.called_weights = True

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    """Get the initial state and side-effect sampling of stochastic weights."""
    if self.built:
      self.call_weights()
    return super(LSTMCellReparameterization, self).get_initial_state(
        inputs=inputs, batch_size=batch_size, dtype=dtype)


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
    input_dim = tf.shape(self.kernel)[0]
    self.sign_input = 2 * tf.random.uniform(
        [batch_size, 4 * input_dim], minval=0, maxval=2, dtype=dtype) - 1
    self.sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1
    self.recurrent_sign_input = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1
    self.recurrent_sign_output = 2 * tf.random.uniform(
        [batch_size, 4 * self.units], minval=0, maxval=2, dtype=dtype) - 1

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
    si_i, si_f, si_c, si_o = tf.split(self.recurrent_sign_input,
                                      num_or_size_splits=4, axis=1)
    so_i, so_f, so_c, so_o = tf.split(self.recurrent_sign_output,
                                      num_or_size_splits=4, axis=1)
    z0 = (x_i + tf.keras.backend.dot(h_tm1_i, k_i) +
          tf.keras.backend.dot(h_tm1_i * si_i, p_i) * so_i)
    z1 = (x_f + tf.keras.backend.dot(h_tm1_f, k_f) +
          tf.keras.backend.dot(h_tm1_f * si_f, p_f) * so_f)
    z2 = (x_c + tf.keras.backend.dot(h_tm1_c, k_c) +
          tf.keras.backend.dot(h_tm1_c * si_c, p_c) * so_c)
    z3 = (x_o + tf.keras.backend.dot(h_tm1_o, k_o) +
          tf.keras.backend.dot(h_tm1_o * si_o, p_o) * so_o)
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
      si_i, si_f, si_c, si_o = tf.split(self.sign_input,
                                        num_or_size_splits=4, axis=1)
      so_i, so_f, so_c, so_o = tf.split(self.sign_output,
                                        num_or_size_splits=4, axis=1)
      x_i = (tf.keras.backend.dot(inputs_i, k_i) +
             tf.keras.backend.dot(inputs_i * si_i, p_i) * so_i)
      x_f = (tf.keras.backend.dot(inputs_f, k_f) +
             tf.keras.backend.dot(inputs_f * si_f, p_f) * so_f)
      x_c = (tf.keras.backend.dot(inputs_c, k_c) +
             tf.keras.backend.dot(inputs_c * si_c, p_c) * so_c)
      x_o = (tf.keras.backend.dot(inputs_o, k_o) +
             tf.keras.backend.dot(inputs_o * si_o, p_o) * so_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = tf.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = tf.keras.backend.bias_add(x_i, b_i)
        x_f = tf.keras.backend.bias_add(x_f, b_f)
        x_c = tf.keras.backend.bias_add(x_c, b_c)
        x_o = tf.keras.backend.bias_add(x_o, b_o)

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
      z = tf.keras.backend.dot(inputs, kernel_mean)
      z += tf.keras.backend.dot(inputs * self.sign_input,
                                perturbation) * self.sign_output
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 = h_tm1 * rec_dp_mask[0]
      recurrent_kernel_mean = self.recurrent_kernel.distribution.mean()
      perturbation = self.recurrent_kernel - recurrent_kernel_mean
      z += tf.keras.backend.dot(h_tm1, recurrent_kernel_mean)
      z += tf.keras.backend.dot(h_tm1 * self.recurrent_sign_input,
                                perturbation) * self.recurrent_sign_output
      if self.use_bias:
        z = tf.keras.backend.bias_add(z, self.bias)

      z = tf.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)

    h = o * self.activation(c)
    return h, [h, c]
