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
"""Definitions for random feature Gaussian process layer.

## References:

[1]: Ali Rahimi and Benjamin Recht. Random Features for Large-Scale Kernel
     Machines. In _Neural Information Processing Systems_, 2007.
     https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf
"""
import math
import tensorflow.compat.v2 as tf

_SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')


class RandomFeatureGaussianProcess(tf.keras.layers.Layer):
  """Gaussian process layer with random feature approximation.

  During training, the model updates the maximum a posteriori (MAP) logits
  estimates and posterior precision matrix using minibatch statistics. During
  inference, the model divides the MAP logit estimates by the predictive
  standard deviation, which is equivalent to approximating the posterior mean
  of the predictive probability via the mean-field approximation.

  User can specify different types of random features by setting
  `use_custom_random_features=True`, and change the initializer and activations
  of the custom random features. For example:

    Linear Kernel: initializer='Identity', activation=tf.identity
    MLP Kernel: initializer='random_normal', activation=tf.nn.relu
    RBF Kernel: initializer='random_normal', activation=tf.math.cos

  Attributes:
    units: (int) The dimensionality of layer.
    num_inducing: (int) The number of random features for the approximation.
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
               custom_random_features_initializer=None,
               custom_random_features_activation=None,
               l2_regularization=0.,
               gp_cov_likelihood='gaussian',
               dtype=None,
               name='random_feature_gaussian_process',
               **gp_output_kwargs):
    """Initializes a random-feature Gaussian process layer instance.

    Args:
      units: (int) Number of output units.
      num_inducing: (int) Number of random Fourier features used for
        approximating the Gaussian process.
      gp_kernel_type: (string) The type of kernel function to use for Gaussian
        process. Currently default to 'gaussian' which is the Gaussian RBF
        kernel.
      gp_kernel_scale: (float) The length-scale parameter of the a
        shift-invariant kernel function, i.e., for RBF kernel:
        exp(-|x1 - x2|**2 / gp_kernel_scale).
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
      custom_random_features_initializer: (str or callable) Initializer for
        the random features. Default to random normal which approximates a RBF
        kernel function if activation function is cos.
      custom_random_features_activation: (callable) Activation function for the
        random feature layer. Default to cosine which approximates a RBF
        kernel function.
      l2_regularization: (float) The strength of l2 regularization on the output
        weights.
      gp_cov_likelihood: (string) Likelihood to use for computing Laplace
        approximation for covariance matrix. Default to `gaussian`.
      dtype: (tf.DType) Input data type.
      name: (string) Layer name.
      **gp_output_kwargs: Additional keyword arguments to dense output layer.
    """
    super(RandomFeatureGaussianProcess, self).__init__(name=name, dtype=dtype)
    self.units = units
    self.num_inducing = num_inducing

    self.normalize_input = normalize_input
    self.gp_input_scale = 1. / tf.sqrt(gp_kernel_scale)
    self.gp_feature_scale = 2. / tf.sqrt(float(num_inducing))

    self.use_custom_random_features = use_custom_random_features
    self.scale_random_features = scale_random_features
    self.return_random_features = return_random_features

    self.gp_cov_momentum = gp_cov_momentum
    self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
    self.gp_cov_likelihood = gp_cov_likelihood

    # Defines module layers.
    self._input_norm_layer = tf.keras.layers.LayerNormalization()

    if use_custom_random_features:
      # Use classific Random Fourier Feature by default.
      random_features_bias_initializer = tf.random_uniform_initializer(
          minval=0., maxval=2. * math.pi)
      if custom_random_features_initializer is None:
        custom_random_features_initializer = tf.initializers.RandomNormal(
            stddev=1.)
      if custom_random_features_activation is None:
        custom_random_features_activation = tf.math.cos

      self._random_feature = tf.keras.layers.Dense(
          units=self.num_inducing,
          use_bias=True,
          activation=custom_random_features_activation,
          kernel_initializer=custom_random_features_initializer,
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
        momentum=self.gp_cov_momentum,
        ridge_penalty=self.gp_cov_ridge_penalty,
        likelihood=self.gp_cov_likelihood,
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
    # Computes random features.
    gp_inputs = inputs
    if self.normalize_input:
      gp_inputs = self._input_norm_layer(gp_inputs)
    elif self.use_custom_random_features:
      # Supports lengthscale for custom random feature layer by directly
      # rescaling the input.
      gp_input_scale = tf.cast(self.gp_input_scale, inputs.dtype)
      gp_inputs = gp_inputs * gp_input_scale

    gp_feature = self._random_feature(gp_inputs)

    if self.scale_random_features:
      # Scale random feature by 2. / sqrt(num_inducing) following [1].
      # When using GP layer as the output layer of a nerual network,
      # it is recommended to turn this scaling off to prevent it from changing
      # the learning rate to the hidden layers.
      gp_feature_scale = tf.cast(self.gp_feature_scale, inputs.dtype)
      gp_feature = gp_feature * gp_feature_scale

    # Computes posterior center (i.e., MAP estimate) and variance.
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
    likelihood: (str) The likelihood to use for computing Laplace approximation
      for the covariance matrix. Can be one of ('binary_logistic', 'poisson',
      'gaussian').
  """

  def __init__(self,
               momentum=0.999,
               ridge_penalty=1e-6,
               likelihood='gaussian',
               dtype=None,
               name='laplace_covariance'):
    if likelihood not in _SUPPORTED_LIKELIHOOD:
      raise ValueError(
          f'"likelihood" must be one of {_SUPPORTED_LIKELIHOOD}, got {likelihood}.'
      )
    self.ridge_penalty = ridge_penalty
    self.momentum = momentum
    self.likelihood = likelihood
    super(LaplaceRandomFeatureCovariance, self).__init__(dtype=dtype, name=name)

  def build(self, input_shape):
    gp_feature_dim = input_shape[-1]

    # Convert gp_feature_dim to int value for TF1 compatibility.
    if isinstance(gp_feature_dim, tf.compat.v1.Dimension):
      gp_feature_dim = gp_feature_dim.value

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

  def make_precision_matrix_update_op(self,
                                      gp_feature,
                                      logits,
                                      precision_matrix,
                                      likelihood='gaussian'):
    """Defines update op for the precision matrix of feature weights."""
    if logits is None and likelihood != 'gaussian':
      raise ValueError(f'"logits" cannot be None when likelihood={likelihood}')

    batch_size = tf.shape(gp_feature)[0]
    batch_size = tf.cast(batch_size, dtype=gp_feature.dtype)

    # Computes batch-specific normalized precision matrix.
    feature_outer_product = tf.matmul(
        gp_feature, gp_feature, transpose_a=True) / batch_size

    if likelihood == 'binary_logistic':
      prob = tf.squeeze(tf.sigmoid(logits))
      precision_matrix_minibatch = feature_outer_product * prob * (1. - prob)
    elif likelihood == 'poisson':
      prob = tf.squeeze(tf.exp(logits))
      precision_matrix_minibatch = feature_outer_product * prob
    else:
      precision_matrix_minibatch = feature_outer_product

    # update population-wise precision matrix
    precision_matrix_new = (
        self.momentum * precision_matrix +
        (1. - self.momentum) * precision_matrix_minibatch)

    # Returns the update op.
    return precision_matrix.assign(precision_matrix_new)

  def compute_predictive_covariance(self, gp_feature):
    """Computes posterior predictive variance.

    Approximates the Gaussian process posterior using random features.
    Given training random feature Phi_tr (num_train, num_hidden) and testing
    random feature Phi_ts (batch_size, num_hidden). The predictive covariance
    matrix is computed as (assuming Gaussian likelihood):

    Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

    where s is the ridge factor to be used for stablizing the inverse, and I is
    the identity matrix with shape (num_hidden, num_hidden).

    Args:
      gp_feature: (tf.Tensor) The random feature of testing data to be used for
        computing the covariance matrix. Shape (batch_size, gp_hidden_size).

    Returns:
      (tf.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
    """
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

  def call(self, inputs, logits=None, training=None):
    """Minibatch updates the GP's posterior precision matrix estimate.

    Args:
      inputs: (tf.Tensor) GP random features, shape (batch_size,
        gp_hidden_size).
      logits: (tf.Tensor) Pre-activation output from the model. Needed
        for Laplace approximation under a non-Gaussian likelihood.
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
          gp_feature=inputs,
          logits=logits,
          precision_matrix=self.precision_matrix)
      self.add_update(precision_matrix_update_op)
      # Return null estimate during training.
      return tf.eye(batch_size, dtype=self.dtype)
    else:
      # Return covariance estimate during inference.
      return self.compute_predictive_covariance(gp_feature=inputs)
