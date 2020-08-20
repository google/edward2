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
"""Definitions for random feature Gaussian process layer."""
import math
import tensorflow.compat.v2 as tf


class RandomFeatureGaussianProcess(tf.keras.layers.Layer):
  """Gaussian process layer with random feature approximation.

  During training, the model updates the maximum a posteriori (MAP) logits
  estimates and posterior precision matrix using minibatch statistics. During
  inference, the model divides the MAP logit estiamtes by the predictive
  standard deviation, which is equivalent to approximating the posterior mean
  of the predictive probability via the mean-field approximation.

  Attributes:
    units: (int) The dimensionality of layer.
    num_inducing: (iny) The number of random features for the approximation.
    is_training: (tf.bool) Whether the layer is set in training mode. If so the
      layer updates the Gaussian process' variance estimate using statistics
      computed from the incoming minibatches.
  """

  def __init__(self,
               units,
               num_inducing=1024,
               gp_kernel_type='gaussian',
               gp_kernel_scale=1.,
               gp_output_bias=0.,
               normalize_input=True,
               gp_kernel_scale_trainable=False,
               gp_output_bias_trainable=False,
               gp_cov_momentum=0.999,
               gp_cov_ridge_penalty=1e-6,
               scale_random_features=True,
               return_random_features=False,
               use_custom_random_features=False,
               custom_random_features_activation=tf.math.cos,
               l2_regularization=0.,
               dtype=None,
               name='random_feature_gaussian_process',
               **gp_output_kwargs):
    """Initializes a Normalized Gaussian Process (NGP) layer instance.

    Args:
      units: (int) Number of output units.
      num_inducing: (int) Number of random Fourier features used for
        approximating the Gaussian process.
      gp_kernel_type: (string) The type of kernel function to use for Gaussian
        process. Currently default to 'gaussian' which is the Gaussian RBF
        kernel.
      gp_kernel_scale: (float) The length-scale parameter of the kernel
        function.
      gp_output_bias: (float) Scalar initial value for the bias vector.
      normalize_input: (bool) Whether to normalize the input to Gaussian
        process.
      gp_kernel_scale_trainable: (bool) Whether the length scale variable is
        trainable.
      gp_output_bias_trainable: (bool) Whether the bias is trainable.
      gp_cov_momentum: (float) A discount factor used to compute the moving
        average for posterior covariance matrix.
      gp_cov_ridge_penalty: (float) Initial Ridge penalty to posterior
        covariance matrix.
      scale_random_features: (bool) Whether to scale the random feature
        by sqrt(2. / num_inducing).
      return_random_features: (bool) Whether to also return random features.
      use_custom_random_features: (bool) Whether to use custom random
        features implemented using tf.keras.layers.Dense.
      custom_random_features_activation: (callable) Activation function for the
        random feature layer. Default to cosine which approximates a RBF
        kernel function.
      l2_regularization: (float) The strength of l2 regularization on the output
        weights.
      dtype: (tf.DType) Input data type.
      name: (string) Layer name.
      **gp_output_kwargs: Additional keyword arguments to dense output layer.
    """
    super(RandomFeatureGaussianProcess, self).__init__(name=name, dtype=dtype)
    self.units = units
    self.num_inducing = num_inducing
    self.normalize_input = normalize_input
    self.scale_random_features = scale_random_features
    self.return_random_features = return_random_features

    # define module layers
    self._input_norm_layer = tf.keras.layers.LayerNormalization()

    if use_custom_random_features:
      random_features_bias_initializer = tf.random_uniform_initializer(
          minval=0., maxval=2. * math.pi)
      self._random_feature = tf.keras.layers.Dense(
          units=self.num_inducing,
          use_bias=True,
          activation=custom_random_features_activation,
          kernel_initializer='random_normal',
          bias_initializer=random_features_bias_initializer,
          trainable=False)
    else:
      self._random_feature = tf.keras.layers.experimental.RandomFourierFeatures(
          output_dim=self.num_inducing,
          kernel_initializer=gp_kernel_type,
          scale=gp_kernel_scale,
          trainable=gp_kernel_scale_trainable,
          dtype=self.dtype)

    self._gp_cov_layer = LaplaceRandomFeatureCovariance(
        momentum=gp_cov_momentum,
        ridge_penalty=gp_cov_ridge_penalty,
        dtype=self.dtype)
    self._gp_output_layer = tf.keras.layers.Dense(
        units=self.units,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l2_regularization),
        dtype=self.dtype,
        **gp_output_kwargs)
    self._gp_output_bias = tf.Variable(
        initial_value=[gp_output_bias] * self.units,
        dtype=self.dtype,
        trainable=gp_output_bias_trainable,
        name='gp_output_bias')

  def call(self, inputs, global_step=None, training=None):
    # define scaling factor
    gp_feature_scale = tf.cast(tf.sqrt(2 / self.num_inducing), inputs.dtype)

    # compute random feature
    gp_inputs = inputs
    if self.normalize_input:
      gp_inputs = self._input_norm_layer(gp_inputs)

    gp_feature = self._random_feature(gp_inputs)
    if self.scale_random_features:
      gp_feature = gp_feature * gp_feature_scale

    # compute posterior center (i.e., MAP estimate) and variance.
    gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias
    gp_covmat = self._gp_cov_layer(gp_feature, training)

    if self.return_random_features:
      return gp_output, gp_covmat, gp_feature
    return gp_output, gp_covmat


class LaplaceRandomFeatureCovariance(tf.keras.layers.Layer):
  """Computes the Gaussian Process covariance using Laplace method.

  At training time, this layer updates the Gaussian process posterior using
  model features in minibatches.

  Attributes:
    momentum: (float) A discount factor used to compute the moving average for
      posterior precision matrix. Analogous to the momentum factor in batch
      normalization.
    ridge_penalty: (float) Initial Ridge penalty to weight covariance matrix.
      This value is used to stablize the eigenvalues of weight covariance
      estimate so that the matrix inverse can be computed for Cov = inv(t(X) * X
      + s * I). The ridge factor s cannot be too large since otherwise it will
      dominate the t(X) * X term and make covariance estimate not meaningful.
  """

  def __init__(self,
               momentum=0.999,
               ridge_penalty=1e-6,
               dtype=None,
               name='laplace_covariance'):
    self.ridge_penalty = ridge_penalty
    self.momentum = momentum
    super(LaplaceRandomFeatureCovariance, self).__init__(dtype=dtype, name=name)

  def build(self, input_shape):
    gp_feature_dim = input_shape[-1]

    # Posterior precision matrix for the GP' random feature coefficients.
    self.precision_matrix = (
        self.add_weight(
            name='gp_precision_matrix',
            shape=(gp_feature_dim, gp_feature_dim),
            dtype=self.dtype,
            initializer=tf.keras.initializers.Identity(self.ridge_penalty),
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA))

    super(LaplaceRandomFeatureCovariance, self).build(input_shape)

  def make_precision_matrix_update_op(self, gp_feature, precision_matrix):
    """Defines update op for the precision matrix of feature weights."""
    batch_size, _ = gp_feature.shape.as_list()

    # compute batch-specific normalized precision matrix.
    precision_matrix_minibatch = tf.matmul(
        gp_feature, gp_feature, transpose_a=True)
    precision_matrix_minibatch = precision_matrix_minibatch / batch_size

    # update population-wise precision matrix
    precision_matrix_new = (
        self.momentum * precision_matrix +
        (1. - self.momentum) * precision_matrix_minibatch)

    # return update op
    return precision_matrix.assign(precision_matrix_new)

  def compute_predictive_covariance(self, gp_feature):
    """Computes posterior predictive variance."""
    # Computes the covariance matrix of the feature coefficient.
    feature_cov_matrix = tf.linalg.inv(self.precision_matrix)

    # Computes the covariance matrix of the gp prediction.
    cov_feature_product = tf.matmul(
        feature_cov_matrix, gp_feature, transpose_b=True)
    gp_cov_matrix = tf.matmul(gp_feature, cov_feature_product)
    return gp_cov_matrix

  def _get_training_value(self, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()

    if isinstance(training, int):
      training = bool(training)

    return training

  def call(self, inputs, training=None):
    """Minibatch updates the GP's posterior precision matrix estimate.

    Args:
      inputs: (tf.Tensor) GP random features, shape (batch_size,
        gp_hidden_size).
      training: (tf.bool) whether or not the layer is in training mode. If in
        training mode, the gp_weight covariance is updated using gp_feature.

    Returns:
      gp_stddev (tf.Tensor): GP posterior predictive variance,
        shape (batch_size, batch_size).
    """
    batch_size = tf.shape(inputs)[0]
    training = self._get_training_value(training)

    if training:
      # Define and register the update op for feature precision matrix.
      precision_matrix_update_op = self.make_precision_matrix_update_op(
          gp_feature=inputs, precision_matrix=self.precision_matrix)
      self.add_update(precision_matrix_update_op)
      # Return null estimate during training.
      return tf.eye(batch_size, dtype=self.dtype)
    else:
      # Return covariance estimate during inference.
      return self.compute_predictive_covariance(gp_feature=inputs)
