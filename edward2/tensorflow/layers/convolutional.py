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

"""Uncertainty-based convolutional layers."""

import functools
from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import initializers
from edward2.tensorflow import random_variable
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow as tf


@utils.add_weight
class Conv2DReparameterization(tf.keras.layers.Conv2D):
  """2D convolution layer (e.g. spatial convolution over images).

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers,

  ```
  p(outputs | inputs) = int conv2d(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zeros',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv2DReparameterization, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super(Conv2DReparameterization, self).call(*args, **kwargs)


@utils.add_weight
class Conv1DReparameterization(tf.keras.layers.Conv1D):
  """1D convolution layer (e.g. temporal convolution over sequences).

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers,

  ```
  p(outputs | inputs) = int conv1d(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format=None,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zeros',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv1DReparameterization, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.kernel_initializer, tf.keras.layers.Layer):
      self.kernel = self.kernel_initializer(self.kernel.shape, self.dtype)
    if isinstance(self.bias_initializer, tf.keras.layers.Layer):
      self.bias = self.bias_initializer(self.bias.shape, self.dtype)

  def call(self, *args, **kwargs):
    self.call_weights()
    kwargs.pop('training', None)
    return super(Conv1DReparameterization, self).call(*args, **kwargs)


class Conv2DFlipout(Conv2DReparameterization):
  """2D convolution layer (e.g. spatial convolution over images).

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers,

  ```
  p(outputs | inputs) = int conv2d(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` random variable must be able
  to decompose as a sum of its mean and a perturbation distribution; the
  perturbation distribution must be independent across weight elements and
  symmetric around zero (for example, a fully factorized Gaussian).
  """

  def call(self, inputs):
    if not isinstance(self.kernel, random_variable.RandomVariable):
      return super(Conv2DFlipout, self).call(inputs)
    self.call_weights()
    outputs = self._apply_kernel(inputs)
    if self.use_bias:
      if self.data_format == 'channels_first':
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
      else:
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def _apply_kernel(self, inputs):
    input_shape = tf.shape(inputs)
    batch_dim = input_shape[0]
    if self._convolution_op is None:
      padding = self.padding
      if self.padding == 'causal':
        padding = 'valid'
      if not isinstance(padding, (list, tuple)):
        padding = padding.upper()
      self._convolution_op = functools.partial(
          tf.nn.convolution,
          strides=self.strides,
          padding=padding,
          data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW',
          dilations=self.dilation_rate)

    if self.data_format == 'channels_first':
      channels = input_shape[1]
      sign_input_shape = [batch_dim, channels, 1, 1]
      sign_output_shape = [batch_dim, self.filters, 1, 1]
    else:
      channels = input_shape[-1]
      sign_input_shape = [batch_dim, 1, 1, channels]
      sign_output_shape = [batch_dim, 1, 1, self.filters]
    sign_input = tf.cast(2 * tf.random.uniform(sign_input_shape,
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32) - 1,
                         inputs.dtype)
    sign_output = tf.cast(2 * tf.random.uniform(sign_output_shape,
                                                minval=0,
                                                maxval=2,
                                                dtype=tf.int32) - 1,
                          inputs.dtype)
    kernel_mean = self.kernel.distribution.mean()
    perturbation = self.kernel - kernel_mean
    outputs = self._convolution_op(inputs, kernel_mean)
    outputs += self._convolution_op(inputs * sign_input,
                                    perturbation) * sign_output
    return outputs


class Conv1DFlipout(Conv1DReparameterization):
  """1D convolution layer (e.g. temporal convolution over sequences).

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers,

  ```
  p(outputs | inputs) = int conv1d(inputs; weights, bias) p(weights, bias)
    dweights dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational
  minimum description length, i.e., it minimizes an upper bound to the negative
  marginal likelihood.

  This layer uses the Flipout estimator (Wen et al., 2018) for integrating with
  respect to the `kernel`. Namely, it applies
  pseudo-independent weight perturbations via independent sign flips for each
  example, enabling variance reduction over independent weight perturbations.
  For this estimator to work, the `kernel` random variable must be able
  to decompose as a sum of its mean and a perturbation distribution; the
  perturbation distribution must be independent across weight elements and
  symmetric around zero (for example, a fully factorized Gaussian).
  """

  def call(self, inputs):
    if not isinstance(self.kernel, random_variable.RandomVariable):
      return super(Conv1DFlipout, self).call(inputs)
    self.call_weights()
    outputs = self._apply_kernel(inputs)
    if self.use_bias:
      if self.data_format == 'channels_first':
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCW')
      else:
        outputs = tf.nn.bias_add(outputs, self.bias, data_format='NWC')
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def _apply_kernel(self, inputs):
    input_shape = tf.shape(inputs)
    batch_dim = input_shape[0]
    if self._convolution_op is None:
      padding = self.padding
      if self.padding == 'causal':
        padding = 'valid'
      if not isinstance(padding, (list, tuple)):
        padding = padding.upper()
      self._convolution_op = functools.partial(
          tf.nn.convolution,
          strides=self.strides,
          padding=padding,
          data_format='NWC' if self.data_format == 'channels_last' else 'NCW',
          dilations=self.dilation_rate)

    if self.data_format == 'channels_first':
      channels = input_shape[1]
      sign_input_shape = [batch_dim, channels, 1]
      sign_output_shape = [batch_dim, self.filters, 1]
    else:
      channels = input_shape[-1]
      sign_input_shape = [batch_dim, 1, channels]
      sign_output_shape = [batch_dim, 1, self.filters]
    sign_input = tf.cast(
        2 * tf.random.uniform(
            sign_input_shape, minval=0, maxval=2, dtype=tf.int32) - 1,
        inputs.dtype)
    sign_output = tf.cast(
        2 * tf.random.uniform(
            sign_output_shape, minval=0, maxval=2, dtype=tf.int32) - 1,
        inputs.dtype)
    kernel_mean = self.kernel.distribution.mean()
    perturbation = self.kernel - kernel_mean
    outputs = self._convolution_op(inputs, kernel_mean)
    outputs += self._convolution_op(inputs * sign_input,
                                    perturbation) * sign_output
    return outputs


class Conv2DHierarchical(Conv2DFlipout):
  """2D convolution layer with hierarchical distributions.

  The layer computes a variational Bayesian approximation to the distribution
  over convolutional layers, and where the distribution over weights
  involves a hierarchical distribution with hidden unit noise coupling vectors
  of the kernel weight matrix (Louizos et al., 2017),

  ```
  p(outputs | inputs) = int conv2d(inputs; new_kernel, bias) p(kernel,
    local_scales, global_scale, bias) dkernel dlocal_scales dglobal_scale dbias.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the kernel and bias. The kernel is written in non-centered
  parameterization where

  ```
  new_kernel[i, j] = kernel[i, j] * local_scale[j] * global_scale.
  ```

  That is, there is "local" multiplicative noise which couples weights for each
  output filter. There is also a "global" multiplicative noise which couples the
  entire weight matrix. By default, the weights are normally distributed and the
  local and global noises are half-Cauchy distributed; this makes the kernel a
  horseshoe distribution (Carvalho et al., 2009; Polson and Scott, 2012).

  The estimation uses Flipout for variance reduction with respect to sampling
  the full weights. Gradients with respect to the distributions' learnable
  parameters backpropagate via reparameterization. Minimizing cross-entropy
  plus the layer's losses performs variational minimum description length,
  i.e., it minimizes an upper bound to the negative marginal likelihood.
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zeros',
               local_scale_initializer='trainable_half_cauchy',
               global_scale_initializer='trainable_half_cauchy',
               kernel_regularizer='normal_kl_divergence',
               bias_regularizer=None,
               local_scale_regularizer='half_cauchy_kl_divergence',
               global_scale_regularizer=regularizers.HalfCauchyKLDivergence(
                   scale=1e-5),
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               local_scale_constraint='softplus',
               global_scale_constraint='softplus',
               **kwargs):
    self.local_scale_initializer = initializers.get(local_scale_initializer)
    self.global_scale_initializer = initializers.get(global_scale_initializer)
    self.local_scale_regularizer = regularizers.get(local_scale_regularizer)
    self.global_scale_regularizer = regularizers.get(global_scale_regularizer)
    self.local_scale_constraint = constraints.get(local_scale_constraint)
    self.global_scale_constraint = constraints.get(global_scale_constraint)
    super(Conv2DHierarchical, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def build(self, input_shape):
    self.local_scale = self.add_weight(
        shape=(self.filters,),
        name='local_scale',
        initializer=self.local_scale_initializer,
        regularizer=self.local_scale_regularizer,
        constraint=self.local_scale_constraint)
    self.global_scale = self.add_weight(
        shape=(),
        name='global_scale',
        initializer=self.global_scale_initializer,
        regularizer=self.global_scale_regularizer,
        constraint=self.global_scale_constraint)
    super(Conv2DHierarchical, self).build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.local_scale_initializer, tf.keras.layers.Layer):
      self.local_scale = self.local_scale_initializer(self.local_scale.shape,
                                                      self.dtype)
    if isinstance(self.global_scale_initializer, tf.keras.layers.Layer):
      self.global_scale = self.global_scale_initializer(self.global_scale.shape,
                                                        self.dtype)
    super(Conv2DHierarchical, self).call_weights()

  def _apply_kernel(self, inputs):
    outputs = super(Conv2DHierarchical, self)._apply_kernel(inputs)
    if self.data_format == 'channels_first':
      local_scale = tf.reshape(self.local_scale, [1, -1, 1, 1])
    else:
      local_scale = tf.reshape(self.local_scale, [1, 1, 1, -1])
    # TODO(trandustin): Figure out what to set local/global scales to at test
    # time. Means don't exist for Half-Cauchy approximate posteriors.
    outputs *= local_scale * self.global_scale
    return outputs


class Conv2DVariationalDropout(Conv2DReparameterization):
  """2D convolution layer with variational dropout (Kingma et al., 2015).

  Implementation follows the additive parameterization of
  Molchanov et al. (2017).
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='trainable_normal',
               bias_initializer='zeros',
               kernel_regularizer='log_uniform_kl_divergence',
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv2DVariationalDropout, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)

  def call(self, inputs, training=None):
    if not isinstance(self.kernel, random_variable.RandomVariable):
      return super(Conv2DVariationalDropout, self).call(inputs)
    self.call_weights()
    if training is None:
      training = tf.keras.backend.learning_phase()
    if self._convolution_op is None:
      padding = self.padding
      if self.padding == 'causal':
        padding = 'valid'
      if not isinstance(padding, (list, tuple)):
        padding = padding.upper()
      self._convolution_op = functools.partial(
          tf.nn.convolution,
          strides=self.strides,
          padding=padding,
          data_format='NHWC' if self.data_format == 'channels_last' else 'NCHW',
          dilations=self.dilation_rate)

    def dropped_inputs():
      """Forward pass with dropout."""
      # Clip magnitude of dropout rate, where we get the dropout rate alpha from
      # the additive parameterization (Molchanov et al., 2017): for weight ~
      # Normal(mu, sigma**2), the variance `sigma**2 = alpha * mu**2`.
      mean = self.kernel.distribution.mean()
      log_variance = tf.math.log(self.kernel.distribution.variance())
      log_alpha = log_variance - tf.math.log(tf.square(mean) +
                                             tf.keras.backend.epsilon())
      log_alpha = tf.clip_by_value(log_alpha, -8., 8.)
      log_variance = log_alpha + tf.math.log(tf.square(mean) +
                                             tf.keras.backend.epsilon())

      means = self._convolution_op(inputs, mean)
      stddevs = tf.sqrt(
          self._convolution_op(tf.square(inputs), tf.exp(log_variance)) +
          tf.keras.backend.epsilon())
      if self.use_bias:
        if self.data_format == 'channels_first':
          means = tf.nn.bias_add(means, self.bias, data_format='NCHW')
        else:
          means = tf.nn.bias_add(means, self.bias, data_format='NHWC')
      outputs = generated_random_variables.Normal(loc=means, scale=stddevs)
      if self.activation is not None:
        outputs = self.activation(outputs)
      return outputs

    # Following tf.keras.Dropout, only apply variational dropout if training
    # flag is True.
    training_value = utils.smart_constant_value(training)
    if training_value is not None:
      if training_value:
        return dropped_inputs()
      else:
        return super(Conv2DVariationalDropout, self).call(inputs)
    return tf.cond(
        pred=training,
        true_fn=dropped_inputs,
        false_fn=lambda: super(Conv2DVariationalDropout, self).call(inputs))


class Conv2DBatchEnsemble(tf.keras.layers.Layer):
  """A batch ensemble convolutional layer."""

  def __init__(self,
               filters,
               kernel_size,
               rank=1,
               ensemble_size=4,
               alpha_initializer='ones',
               gamma_initializer='ones',
               strides=(1, 1),
               padding='valid',
               data_format=None,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv2DBatchEnsemble, self).__init__(**kwargs)
    self.rank = rank
    self.ensemble_size = ensemble_size
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.conv2d = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=None,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=None,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=None)
    self.filters = self.conv2d.filters
    self.kernel_size = self.conv2d.kernel_size
    self.data_format = self.conv2d.data_format

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      input_channel = input_shape[1]
    elif self.data_format == 'channels_last':
      input_channel = input_shape[-1]

    if self.rank > 1:
      alpha_shape = [self.rank, self.ensemble_size, input_channel]
      gamma_shape = [self.rank, self.ensemble_size, self.filters]
    else:
      alpha_shape = [self.ensemble_size, input_channel]
      gamma_shape = [self.ensemble_size, self.filters]
    self.alpha = self.add_weight(
        'alpha',
        shape=alpha_shape,
        initializer=self.alpha_initializer,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        'gamma',
        shape=gamma_shape,
        initializer=self.gamma_initializer,
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
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    input_dim = self.alpha.shape[-1]
    batch_size = tf.shape(inputs)[0]
    examples_per_model = batch_size // self.ensemble_size
    # TODO(ywenxu): Merge the following two cases.
    if self.rank > 1:
      # TODO(ywenxu): Check whether the following works in channels_last case.
      axis_change = -1 if self.data_format == 'channels_first' else 2
      alpha = tf.reshape(tf.tile(self.alpha, [1, 1, examples_per_model]),
                         [self.rank, batch_size, input_dim])
      gamma = tf.reshape(tf.tile(self.gamma, [1, 1, examples_per_model]),
                         [self.rank, batch_size, self.filters])

      alpha = tf.expand_dims(alpha, axis=axis_change)
      alpha = tf.expand_dims(alpha, axis=axis_change)
      gamma = tf.expand_dims(gamma, axis=axis_change)
      gamma = tf.expand_dims(gamma, axis=axis_change)

      perturb_inputs = tf.expand_dims(inputs, 0) * alpha
      perturb_inputs = tf.reshape(perturb_inputs, tf.concat(
          [[-1], perturb_inputs.shape[2:]], 0))
      outputs = self.conv2d(perturb_inputs)

      outputs = tf.reshape(outputs, tf.concat(
          [[self.rank, -1], outputs.shape[1:]], 0))
      outputs = tf.reduce_sum(outputs * gamma, axis=0)
    else:
      axis_change = -1 if self.data_format == 'channels_first' else 1
      alpha = tf.reshape(tf.tile(self.alpha, [1, examples_per_model]),
                         [batch_size, input_dim])
      gamma = tf.reshape(tf.tile(self.gamma, [1, examples_per_model]),
                         [batch_size, self.filters])
      alpha = tf.expand_dims(alpha, axis=axis_change)
      alpha = tf.expand_dims(alpha, axis=axis_change)
      gamma = tf.expand_dims(gamma, axis=axis_change)
      gamma = tf.expand_dims(gamma, axis=axis_change)
      outputs = self.conv2d(inputs*alpha) * gamma

    if self.use_bias:
      bias = tf.reshape(tf.tile(self.bias, [1, examples_per_model]),
                        [batch_size, self.filters])
      bias = tf.expand_dims(bias, axis=axis_change)
      bias = tf.expand_dims(bias, axis=axis_change)
      outputs += bias

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return self.conv2d.compute_output_shape(input_shape)

  def get_config(self):
    config = {
        'ensemble_size': self.ensemble_size,
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
    }
    new_config = super(Conv2DBatchEnsemble, self).get_config()
    new_config.update(self.conv2d.get_config())
    new_config.update(config)
    return new_config


class Conv1DBatchEnsemble(tf.keras.layers.Layer):
  """A batch ensemble convolutional layer."""

  def __init__(self,
               filters,
               kernel_size,
               ensemble_size=4,
               alpha_initializer='ones',
               gamma_initializer='ones',
               strides=1,
               padding='valid',
               data_format='channels_last',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv1DBatchEnsemble, self).__init__(**kwargs)
    self.ensemble_size = ensemble_size
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.conv1d = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=None,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=None,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=None)
    self.filters = self.conv1d.filters
    self.kernel_size = self.conv1d.kernel_size
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
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        'gamma',
        shape=[self.ensemble_size, self.filters],
        initializer=self.gamma_initializer,
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
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    axis_change = -1 if self.data_format == 'channels_first' else 1
    batch_size = tf.shape(inputs)[0]
    input_dim = self.alpha.shape[-1]
    examples_per_model = batch_size // self.ensemble_size
    alpha = tf.reshape(tf.tile(self.alpha, [1, examples_per_model]),
                       [batch_size, input_dim])
    gamma = tf.reshape(tf.tile(self.gamma, [1, examples_per_model]),
                       [batch_size, self.filters])
    alpha = tf.expand_dims(alpha, axis=axis_change)
    gamma = tf.expand_dims(gamma, axis=axis_change)
    outputs = self.conv1d(inputs*alpha) * gamma

    if self.use_bias:
      bias = tf.reshape(tf.tile(self.bias, [1, examples_per_model]),
                        [batch_size, self.filters])
      bias = tf.expand_dims(bias, axis=axis_change)
      outputs += bias

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return self.conv1d.compute_output_shape(input_shape)

  def get_config(self):
    config = {
        'ensemble_size': self.ensemble_size,
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
    }
    new_config = super(Conv1DBatchEnsemble, self).get_config()
    new_config.update(self.conv1d.get_config())
    new_config.update(config)
    return new_config


@utils.add_weight
class CondConv2D(tf.keras.layers.Conv2D):
  """2D conditional convolution layer (e.g. spatial convolution over images).

  This layer extends the base 2D convolution layer to compute example-dependent
  parameters. A CondConv2D layer has 'num_experts` kernels and biases. It
  computes a kernel and bias for each example as a weighted sum of experts
  using the input example-dependent routing weights, then applies the 2D
  convolution to each example.

  Attributes:
    filters: Integer, the dimensionality of the output space (i.e. the number of
      output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    num_experts: The number of expert kernels and biases in the CondConv layer.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying the
      dilation rate to use for dilated convolution. Can be a single integer to
      specify the same value for all spatial dimensions. Currently, specifying
      any `dilation_rate` value != 1 is incompatible with specifying any stride
      value != 1.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to the `kernel` weights
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    4D tensor with shape: `(samples, channels, rows, cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, rows, cols, channels)` if
      data_format='channels_last'.

  Output shape:
    4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
      data_format='channels_first'
    or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
      data_format='channels_last'. `rows` and `cols` values might have changed
      due to padding.
  """

  def __init__(self,
               filters,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(CondConv2D, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) != 4:
      raise ValueError(
          'Inputs to `CondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._get_channel_axis()
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])

    self.kernel_shape = self.kernel_size + (input_dim, self.filters)
    kernel_num_params = 1
    for kernel_dim in self.kernel_shape:
      kernel_num_params *= kernel_dim
    condconv_kernel_shape = (self.num_experts, kernel_num_params)
    self.condconv_kernel = self.add_weight(
        name='condconv_kernel',
        shape=condconv_kernel_shape,
        initializer=initializers.get_condconv_initializer(
            self.kernel_initializer,
            self.num_experts,
            self.kernel_shape),
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias_shape = (self.filters,)
      condconv_bias_shape = (self.num_experts, self.filters)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=initializers.get_condconv_initializer(
              self.bias_initializer,
              self.num_experts,
              self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    self.input_spec = tf.keras.layers.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_dim})

    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent kernels
    kernels = tf.matmul(routing_weights, self.condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    kernels = tf.split(kernels, batch_size, 0)
    # Apply example-dependent convolution to each example in the batch
    outputs_list = []
    # TODO(ywenxu): Check out tf.vectorized_map.
    for input_tensor, kernel in zip(inputs, kernels):
      kernel = tf.reshape(kernel, self.kernel_shape)
      outputs_list.append(
          tf.nn.convolution(
              input_tensor,
              kernel,
              strides=self.strides,
              padding=self._get_padding_op(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def get_config(self):
    config = {'num_experts': self.num_experts}
    base_config = super(CondConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _get_channel_axis(self):
    if self.data_format == 'channels_first':
      return 1
    else:
      return -1

  def _get_padding_op(self):
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    if not isinstance(op_padding, (list, tuple)):
      op_padding = op_padding.upper()
    return op_padding


@utils.add_weight
class DepthwiseCondConv2D(tf.keras.layers.DepthwiseConv2D):
  """Depthwise separable 2D conditional convolution layer.

  This layer extends the base depthwise 2D convolution layer to compute
  example-dependent parameters. A DepthwiseCondConv2D layer has 'num_experts`
  kernels and biases. It computes a kernel and bias for each example as a
  weighted sum of experts using the input example-dependent routing weights,
  then applies the depthwise convolution to each example.

  Attributes:
    kernel_size: An integer or tuple/list of 2 integers, specifying the height
      and width of the 2D convolution window. Can be a single integer to specify
      the same value for all spatial dimensions.
    num_experts: The number of expert kernels and biases in the
      DepthwiseCondConv2D layer.
    strides: An integer or tuple/list of 2 integers, specifying the strides of
      the convolution along the height and width. Can be a single integer to
      specify the same value for all spatial dimensions. Specifying any stride
      value != 1 is incompatible with specifying any `dilation_rate` value != 1.
    padding: one of `'valid'` or `'same'` (case-insensitive).
    depth_multiplier: The number of depthwise convolution output channels for
      each input channel. The total number of depthwise convolution output
      channels will be equal to `filters_in * depth_multiplier`.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs. `channels_last` corresponds
      to inputs with shape `(batch, height, width, channels)` while
      `channels_first` corresponds to inputs with shape `(batch, channels,
      height, width)`. It defaults to the `image_data_format` value found in
      your Keras config file at `~/.keras/keras.json`. If you never set it, then
      it will be 'channels_last'.
    activation: Activation function to use. If you don't specify anything, no
      activation is applied
      (ie. 'linear' activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to the depthwise kernel
      matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the
      layer (its 'activation').
    depthwise_constraint: Constraint function applied to the depthwise kernel
      matrix.
    bias_constraint: Constraint function applied to the bias vector.

  Input shape:
    4D tensor with shape: `[batch, channels, rows, cols]` if
      data_format='channels_first'
    or 4D tensor with shape: `[batch, rows, cols, channels]` if
      data_format='channels_last'.

  Output shape:
    4D tensor with shape: `[batch, filters, new_rows, new_cols]` if
      data_format='channels_first'
    or 4D tensor with shape: `[batch, new_rows, new_cols, filters]` if
      data_format='channels_last'. `rows` and `cols` values might have changed
      due to padding.
  """

  def __init__(self,
               kernel_size,
               num_experts,
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseCondConv2D, self).__init__(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=bias_initializer,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
    if num_experts < 1:
      raise ValueError('A CondConv layer must have at least one expert.')
    self.num_experts = num_experts
    if self.data_format == 'channels_first':
      self.converted_data_format = 'NCHW'
    else:
      self.converted_data_format = 'NHWC'

  def build(self, input_shape):
    if len(input_shape) < 4:
      raise ValueError(
          'Inputs to `DepthwiseCondConv2D` should have rank 4. '
          'Received input shape:', str(input_shape))
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = 3
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv2D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                   input_dim, self.depth_multiplier)

    depthwise_kernel_num_params = 1
    for dim in self.depthwise_kernel_shape:
      depthwise_kernel_num_params *= dim
    depthwise_condconv_kernel_shape = (self.num_experts,
                                       depthwise_kernel_num_params)

    self.depthwise_condconv_kernel = self.add_weight(
        shape=depthwise_condconv_kernel_shape,
        initializer=initializers.get_condconv_initializer(
            self.depthwise_initializer,
            self.num_experts,
            self.depthwise_kernel_shape),
        name='depthwise_condconv_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True)

    if self.use_bias:
      bias_dim = input_dim * self.depth_multiplier
      self.bias_shape = (bias_dim,)
      condconv_bias_shape = (self.num_experts, bias_dim)
      self.condconv_bias = self.add_weight(
          name='condconv_bias',
          shape=condconv_bias_shape,
          initializer=initializers.get_condconv_initializer(
              self.bias_initializer,
              self.num_experts,
              self.bias_shape),
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=4, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs, routing_weights):
    # Compute example dependent depthwise kernels
    depthwise_kernels = tf.matmul(routing_weights,
                                  self.depthwise_condconv_kernel)
    batch_size = inputs.shape[0].value
    inputs = tf.split(inputs, batch_size, 0)
    depthwise_kernels = tf.split(depthwise_kernels, batch_size, 0)
    # Apply example-dependent depthwise convolution to each example in the batch
    outputs_list = []
    if self.data_format == 'channels_first':
      converted_strides = (1, 1) + self.strides
    else:
      converted_strides = (1,) + self.strides + (1,)
    for input_tensor, depthwise_kernel in zip(inputs, depthwise_kernels):
      depthwise_kernel = tf.reshape(depthwise_kernel,
                                    self.depthwise_kernel_shape)
      outputs_list.append(
          tf.nn.depthwise_conv2d(
              input_tensor,
              depthwise_kernel,
              strides=converted_strides,
              padding=self.padding.upper(),
              dilations=self.dilation_rate,
              data_format=self.converted_data_format))
    outputs = tf.concat(outputs_list, 0)

    if self.use_bias:
      # Compute example-dependent biases
      biases = tf.matmul(routing_weights, self.condconv_bias)
      outputs = tf.split(outputs, batch_size, 0)
      biases = tf.split(biases, batch_size, 0)
      # Add example-dependent bias to each example in the batch
      bias_outputs_list = []
      for output, bias in zip(outputs, biases):
        bias = tf.squeeze(bias, axis=0)
        bias_outputs_list.append(
            tf.nn.bias_add(output, bias,
                           data_format=self.converted_data_format))
      outputs = tf.concat(bias_outputs_list, 0)

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  def get_config(self):
    config = {'num_experts': self.num_experts}
    base_config = super(DepthwiseCondConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class DepthwiseConv2DBatchEnsemble(tf.keras.layers.Layer):
  """Batch ensemble of depthwise separable 2D convolutions."""

  def __init__(self,
               kernel_size,
               ensemble_size=4,
               alpha_initializer='ones',
               gamma_initializer='ones',
               strides=(1, 1),
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseConv2DBatchEnsemble, self).__init__(**kwargs)
    self.ensemble_size = ensemble_size
    self.alpha_initializer = initializers.get(alpha_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.bias_constraint = constraints.get(bias_constraint)
    self.activation = tf.keras.activations.get(activation)
    self.use_bias = use_bias
    self.conv2d = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        depth_multiplier=depth_multiplier,
        data_format=data_format,
        activation=None,
        use_bias=False,
        depthwise_initializer=depthwise_initializer,
        bias_initializer=None,
        depthwise_regularizer=depthwise_regularizer,
        bias_regularizer=None,
        activity_regularizer=activity_regularizer,
        depthwise_constraint=depthwise_constraint,
        bias_constraint=None)
    self.kernel_size = self.conv2d.kernel_size
    self.depth_multiplier = self.conv2d.depth_multiplier
    self.data_format = self.conv2d.data_format

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      input_channel = input_shape[1]
    elif self.data_format == 'channels_last':
      input_channel = input_shape[-1]

    filters = input_channel * self.depth_multiplier
    self.alpha = self.add_weight(
        'alpha',
        shape=[self.ensemble_size, input_channel],
        initializer=self.alpha_initializer,
        trainable=True,
        dtype=self.dtype)
    self.gamma = self.add_weight(
        'gamma',
        shape=[self.ensemble_size, filters],
        initializer=self.gamma_initializer,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=[self.ensemble_size, filters],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs):
    axis_change = -1 if self.data_format == 'channels_first' else 1
    batch_size = tf.shape(inputs)[0]
    input_dim = self.alpha.shape[-1]
    filters = self.gamma.shape[-1]
    examples_per_model = batch_size // self.ensemble_size
    alpha = tf.reshape(tf.tile(self.alpha, [1, examples_per_model]),
                       [batch_size, input_dim])
    gamma = tf.reshape(tf.tile(self.gamma, [1, examples_per_model]),
                       [batch_size, filters])
    alpha = tf.expand_dims(alpha, axis=axis_change)
    alpha = tf.expand_dims(alpha, axis=axis_change)
    gamma = tf.expand_dims(gamma, axis=axis_change)
    gamma = tf.expand_dims(gamma, axis=axis_change)
    outputs = self.conv2d(inputs*alpha) * gamma

    if self.use_bias:
      bias = tf.reshape(tf.tile(self.bias, [1, examples_per_model]),
                        [batch_size, filters])
      bias = tf.expand_dims(bias, axis=axis_change)
      bias = tf.expand_dims(bias, axis=axis_change)
      outputs += bias

    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    return self.conv2d.compute_output_shape(input_shape)

  def get_config(self):
    config = {
        'ensemble_size': self.ensemble_size,
        'alpha_initializer': initializers.serialize(self.alpha_initializer),
        'gamma_initializer': initializers.serialize(self.gamma_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'activation': tf.keras.activations.serialize(self.activation),
        'use_bias': self.use_bias,
    }
    new_config = super(DepthwiseConv2DBatchEnsemble, self).get_config()
    new_config.update(self.conv2d.get_config())
    new_config.update(config)
    return new_config
