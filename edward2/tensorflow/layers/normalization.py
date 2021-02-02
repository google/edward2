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

"""Normalization layers.

## References:

[1] Yuichi Yoshida, Takeru Miyato. Spectral Norm Regularization for Improving
    the Generalizability of Deep Learning.
    _arXiv preprint arXiv:1705.10941_, 2017. https://arxiv.org/abs/1705.10941

[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.

[3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368
"""

from edward2.tensorflow import random_variable
from edward2.tensorflow import transformed_random_variable
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1


class ActNorm(tf.python.keras.layers.Layer):
  """Actnorm, an affine reversible layer (Prafulla and Kingma, 2018).

  Weights use data-dependent initialization in which outputs have zero mean
  and unit variance per channel (last dimension). The mean/variance statistics
  are computed from the first batch of inputs.
  """

  def __init__(self, epsilon=tf.python.keras.backend.epsilon(), **kwargs):
    super(ActNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    last_dim = input_shape[-1]
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `ActNorm` '
                       'should be defined. Found `None`.')
    bias = self.add_weight('bias', [last_dim], dtype=self.dtype)
    log_scale = self.add_weight('log_scale', [last_dim], dtype=self.dtype)
    # Set data-dependent initializers.
    bias = bias.assign(self.bias_initial_value)
    with tf.control_dependencies([bias]):
      self.bias = bias
    log_scale = log_scale.assign(self.log_scale_initial_value)
    with tf.control_dependencies([log_scale]):
      self.log_scale = log_scale
    self.built = True

  def __call__(self, inputs, *args, **kwargs):
    if not self.built:
      mean, variance = tf.nn.moments(
          inputs, axes=list(range(inputs.shape.ndims - 1)))
      self.bias_initial_value = -mean
      # TODO(trandustin): Optionally, actnorm multiplies log_scale by a fixed
      # log_scale factor (e.g., 3.) and initializes by
      # initial_value / log_scale_factor.
      self.log_scale_initial_value = tf.math.log(
          1. / (tf.sqrt(variance) + self.epsilon))

    if not isinstance(inputs, random_variable.RandomVariable):
      return super(ActNorm, self).__call__(inputs, *args, **kwargs)
    return transformed_random_variable.TransformedRandomVariable(inputs, self)

  def call(self, inputs):
    return (inputs + self.bias) * tf.exp(self.log_scale)

  def reverse(self, inputs):
    return inputs * tf.exp(-self.log_scale) - self.bias

  def log_det_jacobian(self, inputs):
    """Returns log det | dx / dy | = num_events * sum log | scale |."""
    # Number of events is number of all elements excluding the batch and
    # channel dimensions.
    num_events = tf.reduce_prod(tf.shape(inputs)[1:-1])
    log_det_jacobian = num_events * tf.reduce_sum(self.log_scale)
    return log_det_jacobian


def ensemble_batchnorm(x, ensemble_size=1, use_tpu=True, **kwargs):
  """A modified batch norm layer for Batch Ensemble model.

  Args:
    x: input tensor.
    ensemble_size: number of ensemble members.
    use_tpu: whether the model is running on TPU.
    **kwargs: Keyword arguments to batch normalization layers.

  Returns:
    Output tensor for the block.
  """
  # In BatchEnsemble inference stage, the input to the model is tiled which
  # leads to dynamic shape because of the tf.split function. Such operation
  # is not supported in tf2.0 on TPU. For current workaround, we use single
  # BatchNormalization layer for all ensemble member. This is not correct in
  # math but works in practice.
  if ensemble_size == 1 or use_tpu:
    return tf.python.keras.layers.BatchNormalization(**kwargs)(x)
  name = kwargs.get('name')
  split_inputs = tf.split(x, ensemble_size, axis=0)
  for i in range(ensemble_size):
    if name is not None:
      kwargs['name'] = name + '_{}'.format(i)
    split_inputs[i] = tf.python.keras.layers.BatchNormalization(**kwargs)(
        split_inputs[i])
  return tf.concat(split_inputs, axis=0)


class EnsembleSyncBatchNorm(tf.python.keras.layers.Layer):
  """BatchNorm that averages over ALL replicas. Only works for `NHWC` inputs."""

  def __init__(self, axis=3, ensemble_size=1, momentum=0.99, epsilon=0.001,
               trainable=True, name=None, **kwargs):
    super(EnsembleSyncBatchNorm, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.ensemble_size = ensemble_size

  def build(self, input_shape):
    """Build function."""
    dim = input_shape[-1]
    if self.ensemble_size > 1:
      shape = [self.ensemble_size, dim]
    else:
      shape = [dim]

    self.gamma = self.add_weight(
        name='gamma',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        trainable=True)

    self.beta = self.add_weight(
        name='beta',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        trainable=True)

    self.moving_mean = self.add_weight(
        name='moving_mean',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

    self.moving_variance = self.add_weight(
        name='moving_variance',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

  def _get_mean_and_variance(self, x):
    """Cross-replica mean and variance."""
    replica_context = tf.distribute.get_replica_context()

    if replica_context is not None:
      num_replicas_in_sync = replica_context.num_replicas_in_sync
      if num_replicas_in_sync <= 8:
        group_assignment = None
        num_replicas_per_group = tf.cast(num_replicas_in_sync, tf.float32)
      else:
        num_replicas_per_group = max(8, num_replicas_in_sync // 8)
        group_assignment = np.arange(num_replicas_in_sync, dtype=np.int32)
        group_assignment = group_assignment.reshape(
            [-1, num_replicas_per_group])
        group_assignment = group_assignment.tolist()
        num_replicas_per_group = tf.cast(num_replicas_per_group, tf.float32)

    # This only supports NHWC format.
    if self.ensemble_size > 1:
      height = tf.shape(x)[1]
      width = tf.shape(x)[2]
      input_channels = tf.shape(x)[3]
      x = tf.reshape(x, [self.ensemble_size, -1, height, width, input_channels])
      mean = tf.reduce_mean(x, axis=[1, 2, 3])  # [ensemble_size, channels]
      mean = tf.cast(mean, tf.float32)

      # Var[x] = E[x^2] - E[x]^2
      mean_sq = tf.reduce_mean(tf.square(x), axis=[1, 2, 3])
      mean_sq = tf.cast(mean_sq, tf.float32)
      if replica_context is not None:
        mean = tf1.tpu.cross_replica_sum(mean, group_assignment)
        mean = mean / num_replicas_per_group
        mean_sq = tf1.tpu.cross_replica_sum(mean_sq, group_assignment)
        mean_sq = mean_sq / num_replicas_per_group
      variance = mean_sq - tf.square(mean)
    else:
      mean = tf.reduce_mean(x, axis=[0, 1, 2])
      mean = tf.cast(mean, tf.float32)

      mean_sq = tf.reduce_mean(tf.square(x), axis=[0, 1, 2])
      mean_sq = tf.cast(mean_sq, tf.float32)
      if replica_context is not None:
        mean = tf1.tpu.cross_replica_sum(mean, group_assignment)
        mean = mean / num_replicas_per_group
        mean_sq = tf1.tpu.cross_replica_sum(mean_sq, group_assignment)
        mean_sq = mean_sq / num_replicas_per_group
      variance = mean_sq - tf.square(mean)

    def _assign(moving, normal):
      decay = tf.cast(1. - self.momentum, tf.float32)
      diff = tf.cast(moving, tf.float32) - tf.cast(normal, tf.float32)
      return moving.assign_sub(decay * diff)

    self.add_update(_assign(self.moving_mean, mean))
    self.add_update(_assign(self.moving_variance, variance))

    mean = tf.cast(mean, x.dtype)
    variance = tf.cast(variance, x.dtype)

    return mean, variance

  def call(self, inputs, training=None):
    """Call function."""
    if training:
      mean, variance = self._get_mean_and_variance(inputs)
    else:
      mean, variance = self.moving_mean, self.moving_variance
    if self.ensemble_size > 1:
      batch_size = tf.shape(inputs)[0]
      input_dim = tf.shape(mean)[-1]
      examples_per_model = batch_size // self.ensemble_size
      mean = tf.reshape(tf.tile(mean, [1, examples_per_model]),
                        [batch_size, input_dim])
      variance_epsilon = tf.cast(self.epsilon, variance.dtype)
      inv = tf.math.rsqrt(variance + variance_epsilon)
      if self.gamma is not None:
        inv *= self.gamma
      inv = tf.reshape(tf.tile(inv, [1, examples_per_model]),
                       [batch_size, input_dim])
      # Assuming channel last.
      inv = tf.expand_dims(inv, axis=1)
      inv = tf.expand_dims(inv, axis=1)
      mean = tf.expand_dims(mean, axis=1)
      mean = tf.expand_dims(mean, axis=1)
      if self.beta is not None:
        beta = tf.reshape(tf.tile(self.beta, [1, examples_per_model]),
                          [batch_size, input_dim])
        beta = tf.expand_dims(beta, axis=1)
        beta = tf.expand_dims(beta, axis=1)
      x = inputs * tf.cast(inv, inputs.dtype) + tf.cast(
          beta - mean * inv if self.beta is not None else (
              -mean * inv), inputs.dtype)
    else:
      x = tf.nn.batch_normalization(
          inputs,
          mean=mean,
          variance=variance,
          offset=self.beta,
          scale=self.gamma,
          variance_epsilon=tf.cast(self.epsilon, variance.dtype),
      )
    return x


class SpectralNormalization(tf.python.keras.layers.Wrapper):
  """Implements spectral normalization for Dense layer."""

  def __init__(self,
               layer,
               iteration=1,
               norm_multiplier=0.95,
               training=True,
               aggregation=tf.VariableAggregation.MEAN,
               inhere_layer_name=False,
               **kwargs):
    """Initializer.

    Args:
      layer: (tf.python.keras.layers.Layer) A TF Keras layer to apply normalization to.
      iteration: (int) The number of power iteration to perform to estimate
        weight matrix's singular value.
      norm_multiplier: (float) Multiplicative constant to threshold the
        normalization. Usually under normalization, the singular value will
        converge to this value.
      training: (bool) Whether to perform power iteration to update the singular
        value estimate.
      aggregation: (tf.VariableAggregation) Indicates how a distributed variable
        will be aggregated. Accepted values are constants defined in the class
        tf.VariableAggregation.
      inhere_layer_name: (bool) Whether to inhere the name of the input layer.
      **kwargs: (dict) Other keyword arguments for the layers.Wrapper class.
    """
    self.iteration = iteration
    self.do_power_iteration = training
    self.aggregation = aggregation
    self.norm_multiplier = norm_multiplier

    # Set layer name.
    wrapper_name = kwargs.pop('name', None)
    if inhere_layer_name:
      wrapper_name = layer.name

    if not isinstance(layer, tf.python.keras.layers.Layer):
      raise ValueError('`layer` must be a `tf.python.keras.layer.Layer`. '
                       'Observed `{}`'.format(layer))
    super(SpectralNormalization, self).__init__(
        layer, name=wrapper_name, **kwargs)

  def build(self, input_shape):
    self.layer.build(input_shape)
    self.layer.kernel._aggregation = self.aggregation  # pylint: disable=protected-access
    self._dtype = self.layer.kernel.dtype

    self.w = self.layer.kernel
    self.w_shape = self.w.shape.as_list()
    self.uv_initializer = tf.initializers.random_normal()

    self.v = self.add_weight(
        shape=(1, np.prod(self.w_shape[:-1])),
        initializer=self.uv_initializer,
        trainable=False,
        name='v',
        dtype=self.dtype,
        aggregation=self.aggregation)

    self.u = self.add_weight(
        shape=(1, self.w_shape[-1]),
        initializer=self.uv_initializer,
        trainable=False,
        name='u',
        dtype=self.dtype,
        aggregation=self.aggregation)

    super(SpectralNormalization, self).build()

  def call(self, inputs):
    u_update_op, v_update_op, w_update_op = self.update_weights()
    output = self.layer(inputs)
    w_restore_op = self.restore_weights()

    # Register update ops.
    self.add_update(u_update_op)
    self.add_update(v_update_op)
    self.add_update(w_update_op)
    self.add_update(w_restore_op)

    return output

  def update_weights(self):
    w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

    u_hat = self.u
    v_hat = self.v

    if self.do_power_iteration:
      for _ in range(self.iteration):
        v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
        u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w_reshaped))

    sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
    # Convert sigma from a 1x1 matrix to a scalar.
    sigma = tf.reshape(sigma, [])
    u_update_op = self.u.assign(u_hat)
    v_update_op = self.v.assign(v_hat)

    # Bound spectral norm to be not larger than self.norm_multiplier.
    w_norm = tf.cond((self.norm_multiplier / sigma) < 1, lambda:  # pylint:disable=g-long-lambda
                     (self.norm_multiplier / sigma) * self.w, lambda: self.w)

    w_update_op = self.layer.kernel.assign(w_norm)
    return u_update_op, v_update_op, w_update_op

  def restore_weights(self):
    """Restores layer weights to maintain gradient update (See Alg 1 of [1])."""
    return self.layer.kernel.assign(self.w)


class SpectralNormalizationConv2D(tf.python.keras.layers.Wrapper):
  """Implements spectral normalization for Conv2D layer based on [3]."""

  def __init__(self,
               layer,
               iteration=1,
               norm_multiplier=0.95,
               training=True,
               aggregation=tf.VariableAggregation.MEAN,
               legacy_mode=False,
               **kwargs):
    """Initializer.

    Args:
      layer: (tf.python.keras.layers.Layer) A TF Keras layer to apply normalization to.
      iteration: (int) The number of power iteration to perform to estimate
        weight matrix's singular value.
      norm_multiplier: (float) Multiplicative constant to threshold the
        normalization. Usually under normalization, the singular value will
        converge to this value.
      training: (bool) Whether to perform power iteration to update the singular
        value estimate.
      aggregation: (tf.VariableAggregation) Indicates how a distributed variable
        will be aggregated. Accepted values are constants defined in the class
        tf.VariableAggregation.
      legacy_mode: (bool) Whether to use the legacy implementation where the
        dimension of the u and v vectors are set to the batch size. It should
        not be enabled unless for backward compatibility reasons.
      **kwargs: (dict) Other keyword arguments for the layers.Wrapper class.
    """
    self.iteration = iteration
    self.do_power_iteration = training
    self.aggregation = aggregation
    self.norm_multiplier = norm_multiplier
    self.legacy_mode = legacy_mode

    # Set layer attributes.
    layer._name += '_spec_norm'

    if not isinstance(layer, tf.python.keras.layers.Conv2D):
      raise ValueError(
          'layer must be a `tf.python.keras.layer.Conv2D` instance. You passed: {input}'
          .format(input=layer))
    super(SpectralNormalizationConv2D, self).__init__(layer, **kwargs)

  def build(self, input_shape):
    self.layer.build(input_shape)
    self.layer.kernel._aggregation = self.aggregation  # pylint: disable=protected-access
    self._dtype = self.layer.kernel.dtype

    # Shape (kernel_size_1, kernel_size_2, in_channel, out_channel).
    self.w = self.layer.kernel
    self.w_shape = self.w.shape.as_list()
    self.strides = self.layer.strides

    # Set the dimensions of u and v vectors.
    batch_size = input_shape[0]
    uv_dim = batch_size if self.legacy_mode else 1

    # Resolve shapes.
    in_height = input_shape[1]
    in_width = input_shape[2]
    in_channel = self.w_shape[2]

    out_height = in_height // self.strides[0]
    out_width = in_width // self.strides[1]
    out_channel = self.w_shape[3]

    self.in_shape = (uv_dim, in_height, in_width, in_channel)
    self.out_shape = (uv_dim, out_height, out_width, out_channel)
    self.uv_initializer = tf.initializers.random_normal()

    self.v = self.add_weight(
        shape=self.in_shape,
        initializer=self.uv_initializer,
        trainable=False,
        name='v',
        dtype=self.dtype,
        aggregation=self.aggregation)

    self.u = self.add_weight(
        shape=self.out_shape,
        initializer=self.uv_initializer,
        trainable=False,
        name='u',
        dtype=self.dtype,
        aggregation=self.aggregation)

    super(SpectralNormalizationConv2D, self).build()

  def call(self, inputs):
    u_update_op, v_update_op, w_update_op = self.update_weights()
    output = self.layer(inputs)
    w_restore_op = self.restore_weights()

    # Register update ops.
    self.add_update(u_update_op)
    self.add_update(v_update_op)
    self.add_update(w_update_op)
    self.add_update(w_restore_op)

    return output

  def update_weights(self):
    """Computes power iteration for convolutional filters based on [3]."""
    # Initialize u, v vectors.
    u_hat = self.u
    v_hat = self.v

    if self.do_power_iteration:
      for _ in range(self.iteration):
        # Updates v.
        v_ = tf.nn.conv2d_transpose(
            u_hat,
            self.w,
            output_shape=self.in_shape,
            strides=self.strides,
            padding='SAME')
        v_hat = tf.nn.l2_normalize(tf.reshape(v_, [1, -1]))
        v_hat = tf.reshape(v_hat, v_.shape)

        # Updates u.
        u_ = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding='SAME')
        u_hat = tf.nn.l2_normalize(tf.reshape(u_, [1, -1]))
        u_hat = tf.reshape(u_hat, u_.shape)

    v_w_hat = tf.nn.conv2d(v_hat, self.w, strides=self.strides, padding='SAME')

    sigma = tf.matmul(tf.reshape(v_w_hat, [1, -1]), tf.reshape(u_hat, [-1, 1]))
    # Convert sigma from a 1x1 matrix to a scalar.
    sigma = tf.reshape(sigma, [])

    u_update_op = self.u.assign(u_hat)
    v_update_op = self.v.assign(v_hat)

    w_norm = tf.cond((self.norm_multiplier / sigma) < 1, lambda:      # pylint:disable=g-long-lambda
                     (self.norm_multiplier / sigma) * self.w, lambda: self.w)

    w_update_op = self.layer.kernel.assign(w_norm)

    return u_update_op, v_update_op, w_update_op

  def restore_weights(self):
    """Restores layer weights to maintain gradient update (See Alg 1 of [1])."""
    return self.layer.kernel.assign(self.w)
