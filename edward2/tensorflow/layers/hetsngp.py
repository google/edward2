# coding=utf-8
# Copyright 2022 The Edward2 Authors.
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

"""Library of methods to compute heteroscedastic SNGP predictions."""

from typing import Optional, Union, Callable

from edward2.tensorflow.layers.heteroscedastic import MCSoftmaxDenseFA
from edward2.tensorflow.layers.random_feature import RandomFeatureGaussianProcess
import tensorflow as tf


class HeteroscedasticSNGPLayer(MCSoftmaxDenseFA):
  """MC estimation of softmax approx. to heteroscedastic SNGP predictions."""

  def __init__(
      self,
      num_classes: int,
      num_factors: int = 10,
      temperature: float = 1.0,
      train_mc_samples: int = 1000,
      test_mc_samples: int = 1000,
      compute_pred_variance: bool = False,
      share_samples_across_batch: bool = False,
      logits_only: bool = True,
      eps: float = 1e-7,
      dtype: Optional[tf.dtypes.DType] = None,
      kernel_regularizer: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      bias_regularizer: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      num_inducing: int = 1024,
      gp_kernel_type: str = 'gaussian',
      gp_kernel_scale: float = 1.,
      gp_output_bias: float = 0.,
      normalize_input: bool = False,
      gp_kernel_scale_trainable: bool = False,
      gp_output_bias_trainable: bool = False,
      gp_cov_momentum: float = -1.,
      gp_cov_ridge_penalty: float = 1.,
      scale_random_features: bool = True,
      use_custom_random_features: bool = True,
      custom_random_features_initializer: Optional[Union[str, Callable]] = None,  # pylint: disable=g-bare-generic
      custom_random_features_activation: Optional[Callable] = None,  # pylint: disable=g-bare-generic
      l2_regularization: float = 1e-6,
      gp_cov_likelihood: str = 'gaussian',
      return_gp_cov: bool = True,
      return_random_features: bool = False,
      sngp_var_weight: float = 1.,
      het_var_weight: float = 1.,
      name: str = 'MCSoftmaxSNGP',
      **kwargs):
    """Initializes an MCSoftmaxSNGP layer instance.

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Int. Number of factors for the heteroscedasctic variance.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance. If False the __call__ method will output None for the
        predictive_variance tensor.
      share_samples_across_batch: Boolean. If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference setting = True may solve.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Set True to serialize tf.keras.Sequential models.
      eps: Float. Clip probabilities into [eps, 1.0] before applying log.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
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
      return_gp_cov: (bool) Whether to also return GP covariance matrix.
        If False then no covariance learning is performed.
      return_random_features: (bool) Whether to also return random features.
      sngp_var_weight: (float) Mixing weight for the SNGP variance in the
        total variances during testing.
      het_var_weight: (float) Mixing weight for the heteroscedastic variance
        in the total variance during testing.
      name: (str) The name of the layer used for name scoping.
      **kwargs: Keyword arguments for the SNGP layer.
    """
    super().__init__(num_classes=num_classes,
                     num_factors=num_factors,
                     temperature=temperature,
                     train_mc_samples=train_mc_samples,
                     test_mc_samples=test_mc_samples,
                     compute_pred_variance=compute_pred_variance,
                     share_samples_across_batch=share_samples_across_batch,
                     logits_only=logits_only,
                     eps=eps,
                     dtype=dtype,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     name=name)
    self.sngp_layer = RandomFeatureGaussianProcess(
        units=num_classes,
        num_inducing=num_inducing,
        gp_kernel_type=gp_kernel_type,
        gp_kernel_scale=gp_kernel_scale,
        gp_output_bias=gp_output_bias,
        normalize_input=normalize_input,
        gp_kernel_scale_trainable=gp_kernel_scale_trainable,
        gp_output_bias_trainable=gp_output_bias_trainable,
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        scale_random_features=scale_random_features,
        use_custom_random_features=use_custom_random_features,
        custom_random_features_initializer=custom_random_features_initializer,
        custom_random_features_activation=custom_random_features_activation,
        l2_regularization=l2_regularization,
        gp_cov_likelihood=gp_cov_likelihood,
        return_gp_cov=return_gp_cov,
        return_random_features=return_random_features,
        dtype=dtype,
        name='SNGP_layer',
        **kwargs)
    self.sngp_var_weight = sngp_var_weight
    self.het_var_weight = het_var_weight

  def _compute_loc_param(self, inputs, training):
    """Computes the mean logits as the mean-field logits of the SNGP."""
    return self.sngp_layer(inputs)

  def _compute_scale_param(self, inputs, covmat_sngp, training):
    """Computes the variances for the logits."""
    low_rank, diag = super()._compute_scale_param(inputs)
    sngp_marginal_vars = tf.expand_dims(tf.linalg.diag_part(covmat_sngp), -1)
    if training:
      diag_comp = diag
    else:
      diag_comp = tf.sqrt(self.het_var_weight * tf.square(diag)
                          + self.sngp_var_weight * sngp_marginal_vars)
    return low_rank, diag_comp

  def __call__(self, inputs, training=True, seed=None):
    """Computes predictive and log predictive distribution.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). For multi-class
      classification i.e. num_classes > 2 logits = log_probs and logits can be
      used with the standard tf.nn.sparse_softmax_cross_entropy_with_logits loss
      function. For binary classification i.e. num_classes = 2, logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      tf.nn.sigmoid_cross_entropy_with_logits loss function.

    Raises:
      ValueError if seed is provided but model is running in graph mode.
    """
    # Seed shouldn't be provided in graph mode.
    if not tf.executing_eagerly():
      if seed is not None:
        raise ValueError('Seed should not be provided when running in graph '
                         'mode, but %s was provided.' % seed)
    with tf.name_scope(self._name):
      locs, covmat_sngp = self._compute_loc_param(inputs, training)  # pylint: disable=assignment-from-none
      scale = self._compute_scale_param(inputs, covmat_sngp, training)  # pylint: disable=assignment-from-none

      if training:
        total_mc_samples = self._train_mc_samples
      else:
        total_mc_samples = self._test_mc_samples

      probs_mean, _ = self._compute_predictive_mean(
          locs, scale, total_mc_samples, seed)

      pred_variance = None
      if self._compute_pred_variance:
        pred_variance = self._compute_predictive_variance(
            probs_mean, locs, scale, seed, total_mc_samples)

      probs_mean = tf.clip_by_value(probs_mean, self._eps, 1.0)
      log_probs = tf.math.log(probs_mean)

      if self._num_classes == 2:
        # inverse sigmoid
        probs_mean = tf.clip_by_value(probs_mean, self._eps, 1.0 - self._eps)
        logits = log_probs - tf.math.log(1.0 - probs_mean)
      else:
        logits = log_probs

      if self._logits_only:
        return logits

      return logits, log_probs, probs_mean, pred_variance

  def reset_covariance_matrix(self):
    """Resets the covariance matrix of the SNGP layer."""
    self.sngp_layer.reset_covariance_matrix()
