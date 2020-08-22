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

# Lint as: python3
"""Rank-1 BNN layers."""
import edward2 as ed
import tensorflow as tf


@ed.layers.utils.add_weight
class Conv1DRank1(tf.keras.layers.Layer):
  """A rank-1 priors 1D convolution layer.

  The argument ensemble_size selects the number of mixture components over all
  weights, i.e., an ensemble of size `ensemble_size`. The layer performs a
  forward pass by enumeration, returning a forward pass under each mixture
  component. It takes an input tensor of shape
  [ensemble_size*examples_per_model,] + input_shape and returns an output tensor
  of shape [ensemble_size*examples_per_model,] + output_shape.

  To use a different batch for each mixture, take a minibatch of size
  ensemble_size*examples_per_model. To use the same batch for each mixture, get
  a minibatch of size examples_per_model and tile it by ensemble_size before
  applying any ensemble layers.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               alpha_initializer='trainable_normal',
               gamma_initializer='trainable_normal',
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               alpha_regularizer='normal_kl_divergence',
               gamma_regularizer='normal_kl_divergence',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               alpha_constraint=None,
               gamma_constraint=None,
               kernel_constraint=None,
               bias_constraint=None,
               use_additive_perturbation=False,
               min_perturbation_value=-10,
               max_perturbation_value=10,
               ensemble_size=1,
               **kwargs):
    super().__init__(**kwargs)
    self.filters = filters
    self.kernel_size = kernel_size
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.alpha_initializer = ed.initializers.get(alpha_initializer)
    self.gamma_initializer = ed.initializers.get(gamma_initializer)
    self.bias_initializer = ed.initializers.get(bias_initializer)
    self.alpha_regularizer = ed.regularizers.get(alpha_regularizer)
    self.gamma_regularizer = ed.regularizers.get(gamma_regularizer)
    self.bias_regularizer = ed.regularizers.get(bias_regularizer)
    self.alpha_constraint = ed.constraints.get(alpha_constraint)
    self.gamma_constraint = ed.constraints.get(gamma_constraint)
    self.bias_constraint = ed.constraints.get(bias_constraint)
    self.use_additive_perturbation = use_additive_perturbation
    self.min_perturbation_value = min_perturbation_value
    self.max_perturbation_value = max_perturbation_value
    self.ensemble_size = ensemble_size
    self.conv1d = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint)
    self.data_format = self.conv1d.data_format

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      input_channel = input_shape[1]
    elif self.data_format == 'channels_last':
      input_channel = input_shape[-1]

    self.alpha = self.add_weight(
        'alpha',
        shape=[self.ensemble_size, input_channel],
        initializer=self.alpha_initializer,
        regularizer=self.alpha_regularizer,
        constraint=self.alpha_constraint,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        'gamma',
        shape=[self.ensemble_size, self.filters],
        initializer=self.gamma_initializer,
        regularizer=self.gamma_regularizer,
        constraint=self.gamma_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=[self.ensemble_size, self.filters],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
      self.bias_shape = self.bias.shape
    else:
      self.bias = None
      self.bias_shape = None
    self.alpha_shape = self.alpha.shape
    self.gamma_shape = self.gamma.shape
    self.built = True

  def call(self, inputs):
    axis_change = -1 if self.data_format == 'channels_first' else 1
    batch_size = tf.shape(inputs)[0]
    input_dim = self.alpha_shape[-1]
    examples_per_model = batch_size // self.ensemble_size

    # Sample parameters for each example.
    if isinstance(self.alpha_initializer, tf.keras.layers.Layer):
      alpha = tf.clip_by_value(
          self.alpha_initializer(
              self.alpha_shape,
              self.dtype).distribution.sample(examples_per_model),
          self.min_perturbation_value,
          self.max_perturbation_value)
      alpha = tf.transpose(alpha, [1, 0, 2])
    else:
      alpha = tf.tile(self.alpha, [1, examples_per_model])
    if isinstance(self.gamma_initializer, tf.keras.layers.Layer):
      gamma = tf.clip_by_value(
          self.gamma_initializer(
              self.gamma_shape,
              self.dtype).distribution.sample(examples_per_model),
          self.min_perturbation_value,
          self.max_perturbation_value)
      gamma = tf.transpose(gamma, [1, 0, 2])
    else:
      gamma = tf.tile(self.gamma, [1, examples_per_model])

    alpha = tf.reshape(alpha, [batch_size, input_dim])
    alpha = tf.expand_dims(alpha, axis=axis_change)
    gamma = tf.reshape(gamma, [batch_size, self.filters])
    gamma = tf.expand_dims(gamma, axis=axis_change)

    if self.use_additive_perturbation:
      outputs = self.conv1d(inputs + alpha) + gamma
    else:
      outputs = self.conv1d(inputs * alpha) * gamma

    if self.use_bias:
      if isinstance(self.bias_initializer, tf.keras.layers.Layer):
        bias = self.bias_initializer(
            self.bias_shape, self.dtype).distribution.sample(examples_per_model)
        bias = tf.transpose(bias, [1, 0, 2])
      else:
        bias = tf.tile(self.bias, [1, examples_per_model])
      bias = tf.reshape(bias, [batch_size, self.filters])
      bias = tf.expand_dims(bias, axis=axis_change)
      outputs += bias
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def get_config(self):
    config = {
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
        'alpha_initializer': ed.initializers.serialize(self.alpha_initializer),
        'gamma_initializer': ed.initializers.serialize(self.gamma_initializer),
        'bias_initializer': ed.initializers.serialize(self.bias_initializer),
        'alpha_regularizer': ed.regularizers.serialize(self.alpha_regularizer),
        'gamma_regularizer': ed.regularizers.serialize(self.gamma_regularizer),
        'bias_regularizer': ed.regularizers.serialize(self.bias_regularizer),
        'alpha_constraint': ed.constraints.serialize(self.alpha_constraint),
        'gamma_constraint': ed.constraints.serialize(self.gamma_constraint),
        'bias_constraint': ed.constraints.serialize(self.bias_constraint),
        'use_additive_perturbation': self.use_additive_perturbation,
        'ensemble_size': self.ensemble_size,
    }
    base_config = super().get_config()
    conv_config = self.conv1d.get_config()
    return dict(
        list(base_config.items()) +
        list(conv_config.items()) +
        list(config.items()))


@ed.layers.utils.add_weight
class LSTMCellRank1(tf.keras.layers.LSTMCell):
  """A rank-1 priors LSTM cell.

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
    self.alpha_initializer = ed.initializers.get(alpha_initializer)
    self.gamma_initializer = ed.initializers.get(gamma_initializer)
    self.recurrent_alpha_initializer = ed.initializers.get(
        recurrent_alpha_initializer)
    self.recurrent_gamma_initializer = ed.initializers.get(
        recurrent_gamma_initializer)
    self.alpha_regularizer = ed.regularizers.get(alpha_regularizer)
    self.gamma_regularizer = ed.regularizers.get(gamma_regularizer)
    self.recurrent_alpha_regularizer = ed.regularizers.get(
        recurrent_alpha_regularizer)
    self.recurrent_gamma_regularizer = ed.regularizers.get(
        recurrent_gamma_regularizer)
    self.alpha_constraint = ed.constraints.get(alpha_constraint)
    self.gamma_constraint = ed.constraints.get(gamma_constraint)
    self.recurrent_alpha_constraint = ed.constraints.get(
        recurrent_alpha_constraint)
    self.recurrent_gamma_constraint = ed.constraints.get(
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
        bias_initializer=ed.initializers.get(bias_initializer),
        unit_forget_bias=unit_forget_bias,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=ed.regularizers.get(bias_regularizer),
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=ed.constraints.get(bias_constraint),
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
                                                   tf.keras.layers.Layer)):
        def bias_initializer(_, *args, **kwargs):
          return tf.concat([
              self.bias_initializer([self.ensemble_size, self.units], *args,
                                    **kwargs),
              tf.keras.initializers.Ones()([self.ensemble_size, self.units],
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
      if isinstance(weight_initializer, tf.keras.layers.Layer):
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
        'activation': tf.keras.activations.serialize(self.activation),
        'recurrent_activation': tf.keras.activations.serialize(
            self.recurrent_activation),
        'use_bias': self.use_bias,
        'alpha_initializer': ed.initializers.serialize(self.alpha_initializer),
        'gamma_initializer': ed.initializers.serialize(self.gamma_initializer),
        'kernel_initializer': ed.initializers.serialize(
            self.kernel_initializer),
        'recurrent_alpha_initializer': ed.initializers.serialize(
            self.recurrent_alpha_initializer),
        'recurrent_gamma_initializer': ed.initializers.serialize(
            self.recurrent_gamma_initializer),
        'recurrent_initializer': ed.initializers.serialize(
            self.recurrent_initializer),
        'bias_initializer': ed.initializers.serialize(self.bias_initializer),
        'unit_forget_bias': self.unit_forget_bias,
        'alpha_regularizer': ed.regularizers.serialize(self.alpha_regularizer),
        'gamma_regularizer': ed.regularizers.serialize(self.gamma_regularizer),
        'kernel_regularizer': ed.regularizers.serialize(
            self.kernel_regularizer),
        'recurrent_alpha_regularizer': ed.regularizers.serialize(
            self.recurrent_alpha_regularizer),
        'recurrent_gamma_regularizer': ed.regularizers.serialize(
            self.recurrent_gamma_regularizer),
        'recurrent_regularizer': ed.regularizers.serialize(
            self.recurrent_regularizer),
        'bias_regularizer': ed.regularizers.serialize(self.bias_regularizer),
        'alpha_constraint': ed.constraints.serialize(self.alpha_constraint),
        'gamma_constraint': ed.constraints.serialize(self.gamma_constraint),
        'kernel_constraint': ed.constraints.serialize(self.kernel_constraint),
        'recurrent_alpha_constraint': ed.constraints.serialize(
            self.recurrent_alpha_constraint),
        'recurrent_gamma_constraint': ed.constraints.serialize(
            self.recurrent_gamma_constraint),
        'recurrent_constraint': ed.constraints.serialize(
            self.recurrent_constraint),
        'bias_constraint': ed.constraints.serialize(self.bias_constraint),
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
