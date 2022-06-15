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

"""Library of methods to compute heteroscedastic classification predictions."""

from typing import Callable, Collection, Optional

from edward2.tensorflow.layers import utils
import tensorflow as tf
import tensorflow_probability as tfp

MIN_SCALE_MONTE_CARLO = 1e-3
TEMPERATURE_LOWER_BOUND = 0.3
TEMPERATURE_UPPER_BOUND = 3.0


def compute_temperature(pre_sigmoid_temperature, lower: Optional[float],
                        upper: Optional[float]):
  """Compute the temperature based on the sigmoid parametrization."""
  lower = lower if lower is not None else TEMPERATURE_LOWER_BOUND
  upper = upper if upper is not None else TEMPERATURE_UPPER_BOUND
  temperature = tf.math.sigmoid(pre_sigmoid_temperature)
  return (upper - lower) * temperature + lower


class MCSoftmaxOutputLayerBase(tf.keras.layers.Layer):
  """Base class for MC heteroscesastic output layers.

  Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton and
  Jesse Berent. Correlated Input-Dependent Label Noise in Large-Scale Image
  Classification. In Proc. of the IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR), 2021, pp. 1551-1560.
  https://arxiv.org/abs/2105.10305
  """

  def __init__(self,
               num_classes,
               logit_noise=tfp.distributions.Normal,
               temperature=1.0,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: Optional[float] = None,
               temperature_upper_bound: Optional[float] = None,
               name='MCSoftmaxOutputLayerBase'):
    """Creates an instance of MCSoftmaxOutputLayerBase.

    Args:
      num_classes: Integer. Number of classes for classification task.
      logit_noise: tfp.distributions instance. Must be a location-scale
        distribution. Valid values: tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
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
        due to dynamic shape inference, setting = True may solve.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Useful when a single output Tensor is required e.g.
        tf.keras.Sequential models require a single output Tensor.
      eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.

    Returns:
      MCSoftmaxOutputLayerBase instance.

    Raises:
      ValueError if logit_noise not in tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
    """
    if logit_noise not in (tfp.distributions.Normal,
                           tfp.distributions.Logistic,
                           tfp.distributions.Gumbel):
      raise ValueError('logit_noise must be Normal, Logistic or Gumbel')

    super(MCSoftmaxOutputLayerBase, self).__init__(name=name)

    self._num_classes = num_classes
    self._logit_noise = logit_noise
    self._temperature = temperature
    self._train_mc_samples = train_mc_samples
    self._test_mc_samples = test_mc_samples
    self._compute_pred_variance = compute_pred_variance
    self._share_samples_across_batch = share_samples_across_batch
    self._logits_only = logits_only
    self._eps = eps
    self._return_unaveraged_logits = return_unaveraged_logits
    self._name = name
    self._tune_temperature = tune_temperature
    self._temperature_lower_bound = temperature_lower_bound
    self._temperature_upper_bound = temperature_upper_bound
    if tune_temperature:
      # A zero-initialization means the midpoint of the temperature interval
      # after applying the sigmoid transformation.
      self._pre_sigmoid_temperature = self.add_weight(
          name='pre_sigmoid_temperature',
          trainable=True,
          dtype=tf.float32,
          initializer=tf.zeros_initializer)
    else:
      self._pre_sigmoid_temperature = None

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.

    Args:
      scale: Tensor of shape
        [batch_size, 1 if num_classes == 2 else num_classes].
        Scale parameters of the distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(scale)[0]

    dist = self._logit_noise(
        loc=tf.zeros([num_noise_samples, self._num_classes], dtype=scale.dtype),
        scale=tf.ones([num_noise_samples, self._num_classes],
                      dtype=scale.dtype))

    tf.random.set_seed(seed)
    noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    return tf.transpose(noise_samples, [1, 0, 2]) * tf.expand_dims(scale, 1)

  def _get_temperature(self):
    if self._tune_temperature:
      return compute_temperature(
          self._pre_sigmoid_temperature,
          lower=self._temperature_lower_bound,
          upper=self._temperature_upper_bound)
    else:
      return self._temperature

  def _compute_mc_samples(self, locs, scale, num_samples, seed):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = tf.expand_dims(locs, axis=1)
    noise_samples = self._compute_noise_samples(scale, num_samples, seed)
    latents = locs + noise_samples
    temperature = tf.cast(self._get_temperature(), latents.dtype)
    if self._num_classes == 2:
      return tf.math.sigmoid(latents / temperature)
    else:
      return tf.nn.softmax(latents / temperature)

  def _compute_predictive_mean(self, locs, scale, total_mc_samples, seed):
    """Utility function to compute the estimated predictive distribution.

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      total_mc_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor of shape [batch_size, 1 if num_classes == 2 else num_classes]
      - the mean of the MC samples and Tensor containing the unaveraged samples.
    """
    if self._compute_pred_variance and seed is None:
      seed = utils.gen_int_seed()

    samples = self._compute_mc_samples(locs, scale, total_mc_samples, seed)

    return tf.reduce_mean(samples, axis=1), samples

  def _compute_predictive_variance(self, mean, locs, scale, seed, num_samples):
    """Utility function to compute the per class predictive variance.

    Args:
      mean: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Estimated predictive
        distribution.
      locs: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      seed: Python integer for seeding the random number generator.
      num_samples: Integer. Number of Monte-Carlo samples to take.

    Returns:
      Tensor of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. Estimated predictive variance.
    """
    mean = tf.expand_dims(mean, axis=1)

    mc_samples = self._compute_mc_samples(locs, scale, num_samples, seed)
    total_variance = tf.reduce_mean((mc_samples - mean)**2, axis=1)

    return total_variance

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return

  def call(self, inputs, training=True, seed=None):
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
      locs = self._compute_loc_param(inputs)  # pylint: disable=assignment-from-none
      scale = self._compute_scale_param(inputs)  # pylint: disable=assignment-from-none

      if training:
        total_mc_samples = self._train_mc_samples
      else:
        total_mc_samples = self._test_mc_samples

      probs_mean, samples = self._compute_predictive_mean(
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

      if self._return_unaveraged_logits:
        samples = tf.clip_by_value(samples, self._eps, 1.0)
        samples_log_probs = tf.math.log(samples)

        if self._num_classes == 2:
          # inverse sigmoid
          samples = tf.clip_by_value(samples, self._eps, 1.0 - self._eps)
          samples_logits = samples_log_probs - tf.math.log(1.0 - samples)
        else:
          samples_logits = samples_log_probs

        if self._logits_only:
          return logits, samples_logits

        return logits, samples_logits, log_probs, probs_mean, pred_variance
      elif self._logits_only:
        return logits

      return logits, log_probs, probs_mean, pred_variance

  def get_config(self):
    config = {
        'num_classes': self._num_classes,
        'logit_noise': self._logit_noise,
        'temperature': self._temperature,
        'train_mc_samples': self._train_mc_samples,
        'test_mc_samples': self._test_mc_samples,
        'compute_pred_variance': self._compute_pred_variance,
        'share_samples_across_batch': self._share_samples_across_batch,
        'logits_only': self._logits_only,
        'tune_temperature': self._tune_temperature,
        'temperature_lower_bound': self._temperature_lower_bound,
        'temperature_upper_bound': self._temperature_upper_bound,
        'name': self._name,
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


class MCSoftmaxDense(MCSoftmaxOutputLayerBase):
  """Monte Carlo estimation of softmax approx to heteroscedastic predictions."""

  def __init__(self,
               num_classes,
               logit_noise=tfp.distributions.Normal,
               temperature=1.0,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               dtype=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: Optional[float] = None,
               temperature_upper_bound: Optional[float] = None,
               name='MCSoftmaxDense'):
    """Creates an instance of MCSoftmaxDense.

    This is a MC softmax heteroscedastic drop in replacement for a
    tf.keras.layers.Dense output layer. e.g. simply change:

    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```

    to

    ```python
    logits = MCSoftmaxDense(...)(x)[0]
    ```

    Args:
      num_classes: Integer. Number of classes for classification task.
      logit_noise: tfp.distributions instance. Must be a location-scale
        distribution. Valid values: tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
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
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.

    Returns:
      MCSoftmaxDense instance.

    Raises:
      ValueError if logit_noise not in tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
    """
    assert num_classes >= 2

    super(MCSoftmaxDense, self).__init__(
        num_classes, logit_noise=logit_noise, temperature=temperature,
        train_mc_samples=train_mc_samples, test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only,
        eps=eps,
        return_unaveraged_logits=return_unaveraged_logits,
        tune_temperature=tune_temperature,
        temperature_lower_bound=temperature_lower_bound,
        temperature_upper_bound=temperature_upper_bound,
        name=name)

    self._loc_layer = tf.keras.layers.Dense(
        1 if num_classes == 2 else num_classes, activation=None,
        kernel_regularizer=kernel_regularizer, name='loc_layer', dtype=dtype,
        bias_regularizer=bias_regularizer)
    self._scale_layer = tf.keras.layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation=tf.math.softplus, name='scale_layer', dtype=dtype,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._scale_layer(inputs) + MIN_SCALE_MONTE_CARLO

  def get_config(self):
    config = {
        'loc_layer': tf.keras.layers.serialize(self._loc_layer),
        'scale_layer': tf.keras.layers.serialize(self._scale_layer),
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


class MCSoftmaxDenseFA(MCSoftmaxOutputLayerBase):
  """Softmax and factor analysis approx to heteroscedastic predictions."""

  def __init__(self,
               num_classes,
               num_factors,
               temperature=1.0,
               parameter_efficient=False,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               dtype=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: Optional[float] = None,
               temperature_upper_bound: Optional[float] = None,
               name='MCSoftmaxDenseFA'):
    """Creates an instance of MCSoftmaxDenseFA.

    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    y = softmax(u / temperature)
    ```

    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
    u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
    ```
    where V(x) is a matrix of dimension [num_classes, R=num_factors]
    and d(x) is a vector of dimension [num_classes, 1]
    num_factors << num_classes => approx to sampling ~ N(mu(x), sigma(x))

    This is a MC softmax heteroscedastic drop in replacement for a
    tf.keras.layers.Dense output layer. e.g. simply change:

    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```

    to

    ```python
    logits = MCSoftmaxDenseFA(...)(x)[0]
    ```

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use in approximation to full
        rank covariance matrix.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If false then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
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
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.

    Returns:
      MCSoftmaxDenseFA instance.
    """
    # no need to model correlations between classes in binary case
    assert num_classes > 2
    assert num_factors <= num_classes

    super(MCSoftmaxDenseFA, self).__init__(
        num_classes, logit_noise=tfp.distributions.Normal,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only,
        eps=eps,
        return_unaveraged_logits=return_unaveraged_logits,
        tune_temperature=tune_temperature,
        temperature_lower_bound=temperature_lower_bound,
        temperature_upper_bound=temperature_upper_bound,
        name=name)

    self._num_factors = num_factors
    self._parameter_efficient = parameter_efficient

    if parameter_efficient:
      self._scale_layer_homoscedastic = tf.keras.layers.Dense(
          num_classes, name=name + '_scale_layer_homoscedastic', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)
      self._scale_layer_heteroscedastic = tf.keras.layers.Dense(
          num_classes, name=name + '_scale_layer_heteroscedastic', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)
    else:
      self._scale_layer = tf.keras.layers.Dense(
          num_classes * num_factors, name=name + '_scale_layer', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)

    self._loc_layer = tf.keras.layers.Dense(
        num_classes, name=name + '_loc_layer', dtype=dtype,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)
    self._diag_layer = tf.keras.layers.Dense(
        num_classes, activation=tf.math.softplus, name=name + '_diag_layer',
        dtype=dtype, kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape ([batch_size, num_classes * num_factors],
      [batch_size, num_classes]).
    """
    if self._parameter_efficient:
      return (inputs, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples, seed):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(diag_scale)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_classes],
                     dtype=diag_scale.dtype),
        scale=tf.ones([num_noise_samples, self._num_classes],
                      dtype=diag_scale.dtype))

    tf.random.set_seed(seed)
    diag_noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    diag_noise_samples = tf.transpose(diag_noise_samples, [1, 0, 2])

    return diag_noise_samples * tf.expand_dims(diag_scale, 1)

  def _compute_standard_normal_samples(self, factor_loadings, num_samples,
                                       seed):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(factor_loadings)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_factors],
                     dtype=factor_loadings.dtype),
        scale=tf.ones([num_noise_samples, self._num_factors],
                      dtype=factor_loadings.dtype))

    tf.random.set_seed(seed)
    standard_normal_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    standard_normal_samples = tf.transpose(standard_normal_samples, [1, 0, 2])

    if self._share_samples_across_batch:
      standard_normal_samples = tf.tile(standard_normal_samples,
                                        [tf.shape(factor_loadings)[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    if self._parameter_efficient:
      res = self._scale_layer_homoscedastic(standard_normal_samples)
      res *= tf.expand_dims(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
    else:
      # reshape scale vector into factor loadings matrix
      factor_loadings = tf.reshape(factor_loadings,
                                   [-1, self._num_classes, self._num_factors])

      # transform standard normal into ~ full rank covariance Gaussian samples
      res = tf.einsum('ijk,iak->iaj', factor_loadings, standard_normal_samples)
    return res + diag_noise_samples

  def get_config(self):
    config = {
        'loc_layer': self._loc_layer.get_config(),
        'diag_layer': self._diag_layer.get_config(),
        'num_factors': self._num_factors,
        'parameter_efficient': self._parameter_efficient,
    }

    if self._parameter_efficient:
      config['scale_layer_homoscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_homoscedastic)
      config['scale_layer_heteroscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_heteroscedastic)
    else:
      config['scale_layer'] = tf.keras.layers.serialize(self._scale_layer)

    new_config = super().get_config()
    new_config.update(config)
    return new_config


class MultiHeadMCSoftmaxDenseFA(MCSoftmaxOutputLayerBase):
  """Softmax and factor analysis approx to heteroscedastic predictions.

  Multi Head variation where the output is composed by multiple (ensemble size)
  output predictions, with a shared latent space between ensembles.
  """

  def __init__(self, num_classes, num_factors, ensemble_size, temperature=1.0,
               parameter_efficient=False, train_mc_samples=1000,
               test_mc_samples=1000, compute_pred_variance=False,
               share_samples_across_batch=False, logits_only=False, eps=1e-7,
               dtype=None, kernel_regularizer=None, bias_regularizer=None,
               return_unaveraged_logits=False,
               name='MultiHeadMCSoftmaxDenseFA'):
    """Creates an instance of MultiHeadMCSoftmaxDenseFA.

    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    where x is [x1, x2, ... xn], with n = ensemble_size
    y = [softmax(u_i / temperature)
           for u_i in u.reshape(ensemble_size, num_classes)]
    ```

    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
    u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
    ```
    where V(x) is a matrix of dimension
      [num_classes * ensemble_size, R=num_factors]
    and d(x) is a vector of dimension [num_classes * ensemble_size, 1]
    num_factors << num_classes * ensemble_size => approx to sampling
      ~ N(mu(x), sigma(x))

    This is a MC softmax heteroscedastic drop in replacement for a
    tf.keras.layers.Dense output layer. e.g. simply change:

    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```

    to

    ```python
    logits = MultiHeadMCSoftmaxDenseFA(...)(x)[0]
    ```

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use in approximation to full
        rank covariance matrix.
      ensemble_size: Integer. Size of ensemble.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If false then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
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
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      name: String. The name of the layer used for name scoping.

    Returns:
      MultiHeadMCSoftmaxDenseFA instance.
    """
    # no need to model correlations between classes in binary case
    assert num_classes > 2
    assert num_factors <= num_classes

    super(MultiHeadMCSoftmaxDenseFA, self).__init__(
        num_classes, logit_noise=tfp.distributions.Normal,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only, eps=eps,
        return_unaveraged_logits=return_unaveraged_logits, name=name)

    self._num_factors = num_factors
    self._parameter_efficient = parameter_efficient
    self._ensemble_size = ensemble_size

    if parameter_efficient:
      self._scale_layer_homoscedastic = tf.keras.layers.Dense(
          num_classes * ensemble_size,
          name='scale_layer_homoscedastic', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)
      self._scale_layer_heteroscedastic = tf.keras.layers.Dense(
          num_classes * ensemble_size,
          name='scale_layer_heteroscedastic', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)
    else:
      self._scale_layer = tf.keras.layers.Dense(
          num_classes * ensemble_size * num_factors,
          name='scale_layer', dtype=dtype,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer)

    self._loc_layer = tf.keras.layers.Dense(
        num_classes * ensemble_size, name='loc_layer', dtype=dtype,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)
    self._diag_layer = tf.keras.layers.Dense(
        num_classes * ensemble_size,
        activation=tf.math.softplus, name='diag_layer',
        dtype=dtype, kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def _compute_mc_samples(self, locs, scale, num_samples, seed):
    """Utility function to compute Monte-Carlo samples (using softmax).

    Args:
      locs: Tensor of shape [batch_size, total_mc_samples * ensemble_size,
        1 if num_classes == 2 else num_classes]. Location parameters of the
        distributions to be sampled.
      scale: Tensor of shape [batch_size, total_mc_samples,
        1 if num_classes == 2 else num_classes]. Scale parameters of the
        distributions to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor of shape [batch_size, num_samples,
        1 if num_classes == 2 else num_classes]. All of the MC samples.
    """
    locs = tf.expand_dims(locs, axis=1)
    noise_samples = self._compute_noise_samples(scale, num_samples, seed)
    latents = locs + noise_samples
    latents = tf.keras.layers.Reshape(
        [num_samples, self._ensemble_size, self._num_classes])(latents)
    if self._num_classes == 2:
      return tf.math.sigmoid(latents / self._temperature)
    else:
      return tf.nn.softmax(latents / self._temperature)

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape ([batch_size, num_classes * num_factors],
      [batch_size, num_classes]).
    """
    if self._parameter_efficient:
      return (inputs, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples, seed):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes * ensemble_size].
        Diagonal elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(diag_scale)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples,
                      self._num_classes * self._ensemble_size],
                     dtype=diag_scale.dtype),
        scale=tf.ones([num_noise_samples,
                       self._num_classes * self._ensemble_size],
                      dtype=diag_scale.dtype))

    tf.random.set_seed(seed)
    diag_noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    diag_noise_samples = tf.transpose(diag_noise_samples, [1, 0, 2])

    return diag_noise_samples * tf.expand_dims(diag_scale, 1)

  def _compute_standard_normal_samples(self, factor_loadings, num_samples,
                                       seed):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(factor_loadings)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_factors],
                     dtype=factor_loadings.dtype),
        scale=tf.ones([num_noise_samples, self._num_factors],
                      dtype=factor_loadings.dtype))

    tf.random.set_seed(seed)
    standard_normal_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    standard_normal_samples = tf.transpose(standard_normal_samples, [1, 0, 2])

    if self._share_samples_across_batch:
      standard_normal_samples = tf.tile(standard_normal_samples,
                                        [tf.shape(factor_loadings)[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * ensemble_size * num_factors],
        [batch_size, num_classes * ensemble_size]). Factor loadings and diagonal
          elements for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    if self._parameter_efficient:
      res = self._scale_layer_homoscedastic(standard_normal_samples)
      res *= tf.expand_dims(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
    else:
      # reshape scale vector into factor loadings matrix
      factor_loadings = tf.reshape(
          factor_loadings,
          [-1, self._num_classes * self._ensemble_size, self._num_factors])

      # transform standard normal into ~ full rank covariance Gaussian samples
      res = tf.einsum('ijk,iak->iaj', factor_loadings, standard_normal_samples)
    return res + diag_noise_samples

  def get_config(self):
    config = {
        'loc_layer': self._loc_layer.get_config(),
        'diag_layer': self._diag_layer.get_config(),
        'num_factors': self._num_factors,
        'parameter_efficient': self._parameter_efficient,
        'ensemble_size': self._ensemble_size
    }

    if self._parameter_efficient:
      config['scale_layer_homoscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_homoscedastic)
      config['scale_layer_heteroscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_heteroscedastic)
    else:
      config['scale_layer'] = tf.keras.layers.serialize(self._scale_layer)

    new_config = super().get_config()
    new_config.update(config)
    return new_config


class MCSoftmaxDenseFACustomLayers(MCSoftmaxOutputLayerBase):
  """Softmax and factor analysis approx to heteroscedastic predictions.

  The layers used for the multivariate Normal distributed latent variable are
  given as input to allow custom layers choice.
  """

  def __init__(self, num_classes, num_factors, scale_layer, loc_layer,
               diag_layer, temperature=1.0, parameter_efficient=False,
               train_mc_samples=1000, test_mc_samples=1000,
               compute_pred_variance=False, share_samples_across_batch=False,
               logits_only=False, eps=1e-7, return_unaveraged_logits=False,
               dtype=None, name='MCSoftmaxDenseFACustomLayers'):
    """Creates an instance of MCSoftmaxDenseFACustomLayers.

    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    y = softmax(u / temperature)
    ```

    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
    u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
    ```
    where V(x) is a matrix of dimension [num_classes, R=num_factors]
    and d(x) is a vector of dimension [num_classes, 1]
    num_factors << num_classes => approx to sampling ~ N(mu(x), sigma(x))

    V(x), d(x) and mu(x) are given as input to the layer.

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use in approximation to full
        rank covariance matrix.
      scale_layer: Any layer which takes a [batch_size, d] tensor as input and
        output a [batch_size, num_classes * num_factors] tensor.
      loc_layer: Any layer which takes a [batch_size, d] tensor as input and
        output a [batch_size, num_classes] tensor.
      diag_layer: Any layer which takes a [batch_size, d] tensor as input and
        output a non-negative (>=0) [batch_size, num_classes] tensor.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If false then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
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
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      name: String. The name of the layer used for name scoping.

    Returns:
      MCSoftmaxDenseFACustomLayers instance.
    """
    # no need to model correlations between classes in binary case
    assert num_classes > 2
    assert num_factors <= num_classes

    super(MCSoftmaxDenseFACustomLayers, self).__init__(
        num_classes, logit_noise=tfp.distributions.Normal,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only, eps=eps,
        return_unaveraged_logits=return_unaveraged_logits, name=name)

    self._num_factors = num_factors
    self._parameter_efficient = parameter_efficient

    self._scale_layer = scale_layer
    self._loc_layer = loc_layer
    self._diag_layer = diag_layer

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_classes].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape ([batch_size, num_classes * num_factors],
      [batch_size, num_classes]).
    """
    if self._parameter_efficient:
      return (inputs, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
    else:
      return (self._scale_layer(inputs),
              self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples, seed):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_classes]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(diag_scale)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_classes],
                     dtype=diag_scale.dtype),
        scale=tf.ones([num_noise_samples, self._num_classes],
                      dtype=diag_scale.dtype))

    tf.random.set_seed(seed)
    diag_noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    diag_noise_samples = tf.transpose(diag_noise_samples, [1, 0, 2])

    return diag_noise_samples * tf.expand_dims(diag_scale, 1)

  def _compute_standard_normal_samples(self, factor_loadings, num_samples,
                                       seed):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_classes * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(factor_loadings)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_factors],
                     dtype=factor_loadings.dtype),
        scale=tf.ones([num_noise_samples, self._num_factors],
                      dtype=factor_loadings.dtype))

    tf.random.set_seed(seed)
    standard_normal_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    standard_normal_samples = tf.transpose(standard_normal_samples, [1, 0, 2])

    if self._share_samples_across_batch:
      standard_normal_samples = tf.tile(standard_normal_samples,
                                        [tf.shape(factor_loadings)[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_classes * num_factors],
        [batch_size, num_classes]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    if self._parameter_efficient:
      res = self._scale_layer_homoscedastic(standard_normal_samples)
      res *= tf.expand_dims(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
    else:
      # reshape scale vector into factor loadings matrix
      factor_loadings = tf.reshape(factor_loadings,
                                   [-1, self._num_classes, self._num_factors])

      # transform standard normal into ~ full rank covariance Gaussian samples
      res = tf.einsum('ijk,iak->iaj', factor_loadings, standard_normal_samples)
    return res + diag_noise_samples

  def get_config(self):
    config = {
        'loc_layer': self._loc_layer.get_config(),
        'diag_layer': self._diag_layer.get_config(),
        'num_factors': self._num_factors,
        'parameter_efficient': self._parameter_efficient,
    }

    if self._parameter_efficient:
      config['scale_layer_homoscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_homoscedastic)
      config['scale_layer_heteroscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_heteroscedastic)
    else:
      config['scale_layer'] = tf.keras.layers.serialize(self._scale_layer)

    new_config = super().get_config()
    new_config.update(config)
    return new_config


class MCSigmoidDenseFA(MCSoftmaxOutputLayerBase):
  """Sigmoid and factor analysis approx to heteroscedastic predictions."""

  def __init__(self,
               num_outputs,
               num_factors=0,
               temperature=1.0,
               parameter_efficient=False,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               dtype=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               return_unaveraged_logits=False,
               tune_temperature: bool = False,
               temperature_lower_bound: Optional[float] = None,
               temperature_upper_bound: Optional[float] = None,
               name='MCSigmoidDenseFA'):
    """Creates an instance of MCSigmoidDenseFA.

    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    y = sigmoid(u / temperature)
    ```

    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, identity_K)
    u = mu(x) + matmul(V(x), e) + d(x) * e_d
    ```
    where A(x) is a matrix of dimension [num_outputs, R=num_factors]
    and d(x) is a vector of dimension [num_outputs, 1]
    num_factors << num_outputs => approx to sampling ~ N(mu(x), sigma(x)).

    This is a heteroscedastic drop in replacement for a
    tf.keras.layers.Dense output layer. e.g. simply change:

    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```

    to

    ```python
    logits = MCSigmoidDenseFA(...)(x)[0]
    ```

    Args:
      num_outputs: Integer. Number of outputs.
      num_factors: Integer. Number of factors to use in approximation to full
        rank covariance matrix. If num_factors = 0, then only diagonal
        covariance matrix is used.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If false then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
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
      eps: Float. Clip probabilities into [eps, 1.0 - eps] before applying
        inverse sigmoid.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      return_unaveraged_logits: Boolean. Whether to also return the logits
        before taking the MC average over samples.
      tune_temperature: Boolean. If True, the temperature is optimized during
        the training as any other parameters.
      temperature_lower_bound: Float. The lowest value the temperature can take
        when it is optimized. By default, TEMPERATURE_LOWER_BOUND.
      temperature_upper_bound: Float. The highest value the temperature can take
        when it is optimized. By default, TEMPERATURE_UPPER_BOUND.
      name: String. The name of the layer used for name scoping.

    Returns:
      MCSigmoidDenseFA instance.
    """
    assert num_factors <= num_outputs

    super(MCSigmoidDenseFA, self).__init__(
        2, logit_noise=tfp.distributions.Normal,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only,
        eps=eps,
        return_unaveraged_logits=return_unaveraged_logits,
        tune_temperature=tune_temperature,
        temperature_lower_bound=temperature_lower_bound,
        temperature_upper_bound=temperature_upper_bound,
        name=name)

    self._num_factors = num_factors
    self._parameter_efficient = parameter_efficient
    self._num_outputs = num_outputs

    self._loc_layer = tf.keras.layers.Dense(
        num_outputs, kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer, dtype=dtype, name='loc_layer')

    if num_factors > 0:
      if parameter_efficient:
        self._scale_layer_homoscedastic = tf.keras.layers.Dense(
            num_outputs, name='scale_layer_homoscedastic', dtype=dtype,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)
        self._scale_layer_heteroscedastic = tf.keras.layers.Dense(
            num_outputs, name='scale_layer_heteroscedastic', dtype=dtype,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)
      else:
        self._scale_layer = tf.keras.layers.Dense(
            num_outputs * num_factors, name='scale_layer', dtype=dtype,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)

    self._diag_layer = tf.keras.layers.Dense(
        num_outputs, activation=tf.math.softplus, name='diag_layer',
        bias_initializer='zeros', dtype=dtype,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer)

  def _compute_loc_param(self, inputs):
    """Computes location parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tensor of shape [batch_size, num_outputs].
    """
    return self._loc_layer(inputs)

  def _compute_scale_param(self, inputs):
    """Computes scale parameter of the "logits distribution".

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.

    Returns:
      Tuple of tensors of shape ([batch_size, num_outputs * num_factors],
      [batch_size, num_outputs]).
    """
    if self._num_factors > 0:
      if self._parameter_efficient:
        return (inputs, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
      else:
        return (self._scale_layer(inputs),
                self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)
    else:
      return (None, self._diag_layer(inputs) + MIN_SCALE_MONTE_CARLO)

  def _compute_diagonal_noise_samples(self, diag_scale, num_samples, seed):
    """Compute samples of the diagonal elements logit noise.

    Args:
      diag_scale: `Tensor` of shape [batch_size, num_outputs]. Diagonal
        elements of scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        num_outputs].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(diag_scale)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_outputs], diag_scale.dtype),
        scale=tf.ones([num_noise_samples, self._num_outputs], diag_scale.dtype))

    tf.random.set_seed(seed)
    diag_noise_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    diag_noise_samples = tf.transpose(diag_noise_samples, [1, 0, 2])

    return diag_noise_samples * tf.expand_dims(diag_scale, axis=1)

  def _compute_standard_normal_samples(self, factor_loadings, num_samples,
                                       seed):
    """Utility function to compute samples from a standard normal distribution.

    Args:
      factor_loadings: `Tensor` of shape
        [batch_size, num_outputs * num_factors]. Factor loadings for scale
        parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Samples of shape: [batch_size, num_samples, num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(factor_loadings)[0]

    dist = tfp.distributions.Normal(
        loc=tf.zeros([num_noise_samples, self._num_factors],
                     dtype=factor_loadings.dtype),
        scale=tf.ones([num_noise_samples, self._num_factors],
                      dtype=factor_loadings.dtype))

    tf.random.set_seed(seed)
    standard_normal_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, d], here we reshape to
    # [batch_size, total_mc_samples, d]
    standard_normal_samples = tf.transpose(standard_normal_samples, [1, 0, 2])

    if self._share_samples_across_batch:
      standard_normal_samples = tf.tile(standard_normal_samples,
                                        [tf.shape(factor_loadings)[0], 1, 1])

    return standard_normal_samples

  def _compute_noise_samples(self, scale, num_samples, seed):
    """Utility function to compute the samples of the logit noise.

    Args:
      scale: Tuple of tensors of shape (
        [batch_size, num_outputs * num_factors],
        [batch_size, num_outputs]). Factor loadings and diagonal elements
        for scale parameters of the distribution to be sampled.
      num_samples: Integer. Number of Monte-Carlo samples to take.
      seed: Python integer for seeding the random number generator.

    Returns:
      `Tensor`. Logit noise samples of shape: [batch_size, num_samples,
        num_outputs].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    if self._num_factors == 0:
      return diag_noise_samples

    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    if self._parameter_efficient:
      noise_samples = self._scale_layer_homoscedastic(standard_normal_samples)
      noise_samples *= tf.expand_dims(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
      return noise_samples + diag_noise_samples
    else:
      # reshape scale vector into factor loadings matrix
      factor_loadings = tf.reshape(factor_loadings,
                                   [-1, self._num_outputs, self._num_factors])

      # transform standard normal into ~ full rank covariance Gaussian samples
      noise_samples = tf.einsum('ijk,iak->iaj', factor_loadings,
                                standard_normal_samples)

      return noise_samples + diag_noise_samples

  def get_config(self):
    config = {
        'num_outputs': self._num_outputs,
        'num_factors': self._num_factors,
        'parameter_efficient': self._parameter_efficient,
        'loc_layer': tf.keras.layers.serialize(self._loc_layer),
        'diag_layer': tf.keras.layers.serialize(self._diag_layer),
    }

    if self._parameter_efficient:
      config['scale_layer_homoscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_homoscedastic)
      config['scale_layer_heteroscedastic'] = tf.keras.layers.serialize(
          self._scale_layer_heteroscedastic)
    else:
      config['scale_layer'] = tf.keras.layers.serialize(self._scale_layer)

    new_config = super().get_config()
    new_config.update(config)
    return new_config


class ExactSigmoidDense(tf.keras.layers.Layer):
  """Exact diagonal covariance method for binary/multilabel classification."""

  def __init__(self, num_outputs, logit_noise=tfp.distributions.Normal,
               min_scale=1e-2, logits_only=False, dtype=None,
               name='ExactSigmoidDense'):
    """Creates an instance of ExactSigmoidDense.

    In the case of binary classification or multilabel classification with
    diagonal covariance matrix we can compute the predictive distribution
    exactly. We do not need to make the softmax/sigmoid approximation and we do
    not need to use Monte Carlo estimation.

    This layer is a drop in replacement for a tf.keras.layers.Dense output
    layer for binary and multilabel classification problems, simply change:

    ```python
    logits = tf.keras.layers.Dense(num_outputs, ...)(x)
    ```

    to

    ```python
    logits = ExactSigmoidDense(num_outputs)(x)[0]
    ```

    Args:
      num_outputs: Integer. Number of outputs. If binary classification the
        num_outputs is expected to be 1, for multilabel classification the
        num_outputs is expected to be equal to the number of labels.
      logit_noise: tfp.distributions instance. Must be either
        tfp.distributions.Normal or tfp.distributions.Logistic.
      min_scale: Float. Minimum value for the scale parameter on the
        latent distribution. If experiencing numerical instability during
        training, increasing this value may help.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Set True to serialize tf.keras.Sequential models.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      name: String. The name of the layer used for name scoping.

    Returns:
      ExactSigmoidDense instance.

    Raises:
      ValueError if logit_noise not in tfp.distributions.Normal,
        tfp.distributions.Logistic.
    """
    super(ExactSigmoidDense, self).__init__(name=name)

    if logit_noise not in (tfp.distributions.Normal,
                           tfp.distributions.Logistic):
      raise ValueError('logit_noise must be Normal or Logistic')

    self._loc_layer = tf.keras.layers.Dense(num_outputs, name='loc_layer',
                                            dtype=dtype)

    self._diag_layer = tf.keras.layers.Dense(
        num_outputs, activation=tf.math.softplus, name='diag_layer',
        dtype=dtype)

    self._num_outputs = num_outputs
    self._logit_noise = logit_noise
    self._min_scale = min_scale
    self._logits_only = logits_only
    self._name = name

  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distribution.

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      Tuple of (logits, log_probs, probs). Logits can be used with the
      tf.nn.sigmoid_cross_entropy_with_logits loss function.
    """
    with tf.name_scope(self._name):
      loc = self._loc_layer(inputs)
      scale = self._diag_layer(inputs) + self._min_scale
      loc = tf.cast(loc, tf.float32)
      scale = tf.cast(scale, tf.float32)

      dist = self._logit_noise(loc=tf.constant(0.0, dtype=tf.float32),
                               scale=tf.constant(1.0, dtype=tf.float32))

      if self._logit_noise == tfp.distributions.Normal:
        probs = dist.cdf(loc / scale)  # pylint: disable=protected-access
        log_probs = dist._log_cdf(loc / scale)  # pylint: disable=protected-access

        # inverse sigmoid
        logits = log_probs - dist._log_cdf(-loc / scale)  # pylint: disable=protected-access
      elif self._logit_noise == tfp.distributions.Logistic:
        probs = tf.math.sigmoid(loc / scale)
        log_probs = tf.math.log_sigmoid(loc / scale)
        logits = loc / scale

      if self._logits_only:
        return logits

      return logits, log_probs, probs

  def get_config(self):
    config = {
        'loc_layer': tf.keras.layers.serialize(self._loc_layer),
        'diag_layer': tf.keras.layers.serialize(self._diag_layer),
        'num_outputs': self._num_outputs,
        'logit_noise': self._logit_noise,
        'min_scale': self._min_scale,
        'logits_only': self._logits_only,
        'name': self._name,
    }
    new_config = super().get_config()
    new_config.update(config)
    return new_config


class EnsembleHeteroscedasticOutputs(tf.keras.layers.Layer):
  """Ensembles multiple heteroscedastic output layers."""

  def __init__(self, num_classes, layers, ensemble_weighting,
               averaging='ensemble_cross_ent',
               eps=1e-7, name='HeteroscedasticEnsemble'):
    """Creates an instance of EnsembleHeteroscedasticOutputs.

    Ensembles multiple heteroscedastic output layers. Usage:

    ```python
    layer_1 = ExactSigmoidDense(num_outputs,
                                logit_noise=tfp.distributions.Normal)
    layer_2 = ExactSigmoidDense(num_outputs,
                                logit_noise=tfp.distributions.Logistic)
    logits = EnsembleHeteroscedasticOutputs(
      2, (layer_1, layer_2), (0.5, 0.5))(x)[0]
    ```

    Args:
      num_classes: Integer. Number of classes for classification task.
      layers: Tuple of tf.keras.layers.Layer from heteroscedastic.py.
      ensemble_weighting: Tuple of len(layers) representing a probability
        distribution over layers.
      averaging: String `ensemble_cross_ent` or `gibbs_cross_ent`. For
        `ensemble_cross_ent`: loss = - log (sum_i  weighting[i] * p_i)
        i.e. ensemble members are trained in the knowledge they will be
        ensembled. For `gibbs_cross_ent`:
        loss = - sum_i weighting[i] * log (p_i), this can help promote
        diversity.
      eps: Float. Clip ensemble members probabilities into [eps, 1.0 - eps].
      name: String. The name of the layer used for name scoping.

    Returns:
      Ensemble instance.
    """
    super(EnsembleHeteroscedasticOutputs, self).__init__(name=name)

    if(abs(1.0 - sum(ensemble_weighting)) > 1e-5 or
       any([w < 0.0 for w in ensemble_weighting])):
      raise ValueError(
          'ensemble_weighting must be a valid probability distribution.')

    assert averaging in ('ensemble_cross_ent', 'gibbs_cross_ent')

    self._num_classes = num_classes
    self._layers = layers
    self._num_layers = len(layers)

    self._ensemble_weighting = ensemble_weighting
    self._averaging = averaging
    self._eps = eps
    self._name = name

  def __call__(self, inputs, training=True):
    """Computes predictive and log predictive distribution.

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.

    Returns:
      Tuple of (logits, log_probs, probs). Logits can be used with the
      tf.nn.sigmoid_cross_entropy_with_logits or
      tf.nn.softmax_cross_entropy_with_logits loss functions.
    """
    with tf.name_scope(self._name):
      if self._averaging == 'ensemble_cross_ent':
        log_weights = tf.math.log(self._ensemble_weighting)

        weighted_log_probs = []
        for i, layer in enumerate(self._layers):
          log_weight = log_weights[i]

          member_log_prob = layer(inputs, training)[1]
          log_weight = tf.cast(log_weight, member_log_prob.dtype)
          weighted_log_probs.append(
              tf.cast(log_weight + member_log_prob, tf.float32))

        weighted_log_probs = tf.stack(weighted_log_probs, axis=-1)

        log_probs = tf.math.reduce_logsumexp(weighted_log_probs, axis=-1)

        probs = tf.math.exp(log_probs)
        probs = tf.clip_by_value(probs, self._eps, 1.0 - self._eps)
        if self._num_classes == 2:
          # inverse sigmoid to transform probs to logits for sigmoid cross-ent
          logits = log_probs - tf.math.log(1.0 - probs)
        else:
          logits = log_probs
      else:
        logits = 0.0
        probs = 0.0
        for i, layer in enumerate(self._layers):
          weight = self._ensemble_weighting[i]
          member_results = layer(inputs, training)
          member_logits = member_results[0]
          weight = tf.cast(weight, member_logits.dtype)
          logits += tf.cast(weight * member_logits, tf.float32)
          probs += tf.cast(weight * member_results[2], tf.float32)

        probs = tf.clip_by_value(probs, self._eps, 1.0 - self._eps)
        log_probs = tf.math.log(probs)
        if not training:
          if self._num_classes == 2:
            logits = log_probs - tf.math.log(1.0 - probs)
          else:
            logits = log_probs

      return logits, log_probs, probs


class MCSoftmaxDenseFASegmentation(MCSoftmaxDenseFA):
  """Softmax heteroscedastic layer for 4D inputs."""

  def __init__(self, num_classes: int, num_factors: int,
               temperature: float = 1.0,
               parameter_efficient: bool = False,
               train_mc_samples: int = 1000,
               test_mc_samples: int = 1000,
               compute_pred_variance: bool = False,
               share_samples_across_batch: bool = False,
               logits_only: bool = False,
               eps: float = 1e-7, dtype: Optional[tf.dtypes.DType] = None,
               kernel_regularizer: Optional[
                   Callable[[tf.Tensor], tf.Tensor]] = None,
               bias_regularizer: Optional[
                   Callable[[tf.Tensor], tf.Tensor]] = None,
               name: str = 'MCSoftmaxDenseFASegmentation'):
    """Creates an instance of MCSoftmaxDenseFASegmentation.

    if we assume:
    ```
    u ~ N(mu(x), sigma(x))
    y = softmax(u / temperature)
    ```

    we can do a low rank approximation of sigma(x) the full rank matrix as:
    ```
    eps_R ~ N(0, I_R), eps_K ~ N(0, I_K)
    u = mu(x) + matmul(V(x), eps_R) + d(x) * eps_K
    ```
    where V(x) is a matrix of dimension [num_classes, R=num_factors]
    and d(x) is a vector of dimension [num_classes, 1]
    num_factors << num_classes => approx to sampling ~ N(mu(x), sigma(x))

    This is a MC softmax heteroscedastic drop-in replacement for a
    tf.keras.layers.Dense output layer for 4D inputs. e.g. simply change:

    ```python
    logits = tf.keras.layers.Dense(...)(x)
    ```

    to

    ```python
    logits = MCSoftmaxDenseFASegmentation(...)(x)[0]
    ```

    Args:
      num_classes: Number of classes for classification task.
      num_factors: Number of factors to use in approximation to full
        rank covariance matrix.
      temperature: The softmax temperature.
      parameter_efficient: Whether to use the parameter efficient
        version of the method. If True then samples from the latent distribution
        are generated as: mu(x) + v(x) * matmul(V, eps_R) + diag(d(x), eps_K)),
        where eps_R ~ N(0, I_R), eps_K ~ N(0, I_K). If False then latent samples
        are generated as: mu(x) + matmul(V(x), eps_R) + diag(d(x), eps_K)).
        Computing V(x) as function of x increases the number of parameters
        introduced by the method.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Whether to estimate the predictive
        variance. If False the __call__ method will output None for the
        predictive_variance tensor.
      share_samples_across_batch: If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference setting = True may solve.
      logits_only: If True, only return the logits from the __call__
        method. Set True to serialize tf.keras.Sequential models.
      eps: Clip probabilities into [eps, 1.0] before applying log.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      name: The name of the layer used for name scoping.

    Returns:
      MCSoftmaxDenseFASegmentation instance.
    """
    super(MCSoftmaxDenseFASegmentation, self).__init__(
        num_classes, num_factors, temperature=temperature,
        parameter_efficient=parameter_efficient,
        train_mc_samples=train_mc_samples, test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance,
        share_samples_across_batch=share_samples_across_batch,
        logits_only=logits_only, eps=eps, dtype=None, kernel_regularizer=None,
        bias_regularizer=None, name=name)

  def _genrate_4d_standard_normal_samples(self,
                                          dimensions: Collection[tf.Tensor],
                                          num_samples: int, seed: int,
                                          dtype: tf.dtypes.DType) -> tf.Tensor:
    """Computes samples from a 4 dimensional standard Normal distribution.

    Args:
      dimensions: Collection[Tensor] of length 4 containing the dimensions of
        the Gaussian to be sampled.
      num_samples: Number of Monte-Carlo samples to take.
      seed: Seed for the random number generator.
      dtype: Valid Tensorflow dtype. The dtype of the returned samples.

    Returns:
      Tensor. Standard normal samples of shape: [dimensions[0], num_samples,
        dimensions[1], dimensions[2], dimensions[3]].
    """
    assert len(dimensions) == 4
    dist = tfp.distributions.Normal(
        loc=tf.zeros(dimensions, dtype=dtype),
        scale=tf.ones(dimensions, dtype=dtype))

    tf.random.set_seed(seed)
    standard_normal_samples = dist.sample(num_samples, seed=seed)

    # dist.sample(total_mc_samples) returns Tensor of shape
    # [total_mc_samples, batch_size, ...], here we reshape to
    # [batch_size, total_mc_samples, ...]
    standard_normal_samples = tf.transpose(standard_normal_samples,
                                           [1, 0, 2, 3, 4])
    return standard_normal_samples

  def _compute_diagonal_noise_samples(self, diag_scale: tf.Tensor,
                                      num_samples: int, seed: int) -> tf.Tensor:
    """Computes samples of the diagonal elements logit noise.

    Args:
      diag_scale: Tensor of shape [batch_size, height, width, num_classes].
        Diagonal elements of scale parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.
      seed: Seed for the random number generator.

    Returns:
      Tensor. Logit noise samples of shape: [batch_size, height, width,
        num_samples, 1 if num_classes == 2 else num_classes].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(diag_scale)[0]

    height, width = tf.shape(diag_scale)[1:3]

    diag_noise_samples = self._genrate_4d_standard_normal_samples(
        (num_noise_samples, height, width, self._num_classes), num_samples,
        seed, diag_scale.dtype)

    return diag_noise_samples * tf.expand_dims(diag_scale, 1)

  def _compute_standard_normal_samples(self, factor_loadings: tf.Tensor,
                                       num_samples: int,
                                       seed: int) -> tf.Tensor:
    """Utility that computes samples from a standard normal distribution.

    Args:
      factor_loadings: Tensor of shape
        [batch_size, height, width, num_classes * num_factors]. Factor loadings
        for scale parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.
      seed: Seed for the random number generator.

    Returns:
      Tensor. Samples of shape: [batch_size, num_samples, height, width,
        num_factors].
    """
    if self._share_samples_across_batch:
      num_noise_samples = 1
    else:
      num_noise_samples = tf.shape(factor_loadings)[0]

    height, width = tf.shape(factor_loadings)[1:3]

    standard_normal_samples = self._genrate_4d_standard_normal_samples(
        (num_noise_samples, height, width, self._num_factors), num_samples,
        seed, factor_loadings.dtype)

    if self._share_samples_across_batch:
      return tf.tile(
          standard_normal_samples, [tf.shape(factor_loadings)[0], 1, 1, 1, 1])
    else:
      return standard_normal_samples

  def _compute_noise_samples(self, scale: Collection[tf.Tensor],
                             num_samples: int, seed: int) -> tf.Tensor:
    """Utility function that computes the samples of the logit noise.

    Args:
      scale: Collection[Tensor] of shape (
        [batch_size, height, width, num_classes * num_factors],
        [batch_size, height, width, num_classes]). Factor loadings and diagonal
        elements for scale parameters of the distribution to be sampled.
      num_samples: Number of Monte-Carlo samples to take.
      seed: Seed for the random number generator.

    Returns:
      Tensor. Logit noise samples of shape: [batch_size, num_samples, height,
        width, 1 if num_classes == 2 else num_classes].
    """
    factor_loadings, diag_scale = scale

    # Compute the diagonal noise
    diag_noise_samples = self._compute_diagonal_noise_samples(diag_scale,
                                                              num_samples, seed)

    # Now compute the factors
    standard_normal_samples = self._compute_standard_normal_samples(
        factor_loadings, num_samples, seed)

    if self._parameter_efficient:
      res = self._scale_layer_homoscedastic(standard_normal_samples)
      res *= tf.expand_dims(
          self._scale_layer_heteroscedastic(factor_loadings), 1)
    else:
      # reshape scale vector into factor loadings matrix
      batch_size, height, width = tf.shape(factor_loadings)[:3]
      factor_loadings = tf.reshape(
          factor_loadings,
          [batch_size, height, width, self._num_classes, self._num_factors])

      # transform standard normal into ~ full rank covariance Gaussian samples
      res = tf.einsum('ihwjk,iahwk->iahwj', factor_loadings,
                      standard_normal_samples)
    return res + diag_noise_samples
