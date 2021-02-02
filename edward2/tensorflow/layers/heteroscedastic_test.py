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

"""Tests for heteroscedastic.py."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def test_cases():
  return parameterized.named_parameters(
      {
          'testcase_name': '_MCSoftmaxDense_logit_noise_normal_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'MCSoftmaxDense'
      }, {
          'testcase_name': '_MCSoftmaxDense_logit_noise_logistic_10',
          'logit_noise': tfp.distributions.Logistic,
          'num_classes': 10,
          'model_type': 'MCSoftmaxDense'
      }, {
          'testcase_name': '_MCSoftmaxDense_logit_noise_gumbel_10',
          'logit_noise': tfp.distributions.Gumbel,
          'num_classes': 10,
          'model_type': 'MCSoftmaxDense'
      },
      {
          'testcase_name': '_MCSoftmaxDenseFA_logit_noise_normal_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'MCSoftmaxDenseFA'
      },
      {
          'testcase_name': '_MCSigmoidDenseFA_logit_noise_normal_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'MCSigmoidDenseFA'
      },
      {
          'testcase_name': '_MCSoftmaxDenseFAPE_logit_noise_normal_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'MCSoftmaxDenseFAPE'
      },
      {
          'testcase_name': '_MCSigmoidDenseFAPE_logit_noise_normal_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'MCSigmoidDenseFAPE'
      },
      {
          'testcase_name': '_MCSigmoidDenseFA_logit_noise_normal_2',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 2,
          'model_type': 'MCSigmoidDenseFA'
      }, {
          'testcase_name': '_MCSigmoidDenseFA_logit_noise_logistic_2',
          'logit_noise': tfp.distributions.Logistic,
          'num_classes': 2,
          'model_type': 'MCSigmoidDenseFA'
      }, {
          'testcase_name': '_MCSigmoidDenseFA_logit_noise_gumbel_2',
          'logit_noise': tfp.distributions.Gumbel,
          'num_classes': 2,
          'model_type': 'MCSigmoidDenseFA'
      },
      {
          'testcase_name': '_Exact_logit_noise_normal_2',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 2,
          'model_type': 'Exact'
      }, {
          'testcase_name': '_Exact_logit_noise_logistic_2',
          'logit_noise': tfp.distributions.Logistic,
          'num_classes': 2,
          'model_type': 'Exact'
      },
      {
          'testcase_name': '_EnsembleGibbsCE_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'EnsembleGibbsCE'
      }, {
          'testcase_name': '_EnsembleGibbsCE_2',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 2,
          'model_type': 'EnsembleGibbsCE'
      }, {
          'testcase_name': '_EnsembleEnsembleCE_10',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 10,
          'model_type': 'EnsembleEnsembleCE'
      }, {
          'testcase_name': '_EnsembleEnsembleCE_2',
          'logit_noise': tfp.distributions.Normal,
          'num_classes': 2,
          'model_type': 'EnsembleEnsembleCE'
      },)


class Classifier(tf.python.keras.Model):
  """Wrapper for classifiers defined below.

  Handles different architectures and differences between eager/graph execution.
  """

  def __init__(self, model_type='MCSoftmaxDense', num_classes=2,
               logit_noise=tfp.distributions.Normal,
               **kwargs):
    super().__init__()
    if model_type == 'MCSoftmaxDense':
      self.classifier = DenseClassifier(num_classes, **kwargs)
    elif model_type == 'MCSoftmaxDenseFA':
      self.classifier = DenseFAClassifier(
          num_classes, num_factors=max(num_classes//2, 2), **kwargs)
    elif model_type == 'MCSigmoidDenseFA':
      self.classifier = SigmoidDenseFAClassifier(
          num_classes,
          num_factors=max(num_classes//2, 2) if num_classes > 2 else 0,
          **kwargs)
    elif model_type == 'MCSoftmaxDenseFAPE':
      self.classifier = DenseFAClassifier(
          num_classes, num_factors=max(num_classes//2, 2),
          parameter_efficient=True, **kwargs)
    elif model_type == 'MCSigmoidDenseFAPE':
      self.classifier = SigmoidDenseFAClassifier(
          num_classes, num_factors=max(num_classes//2, 2),
          parameter_efficient=True, **kwargs)
    elif model_type == 'Exact':
      self.classifier = ExactSigmoidDenseClassifier(num_classes, logit_noise)
    elif model_type == 'EnsembleGibbsCE':
      self.classifier = EnsembleClassifier(
          num_classes, averaging='gibbs_cross_ent')
    elif model_type == 'EnsembleEnsembleCE':
      self.classifier = EnsembleClassifier(
          num_classes, averaging='ensemble_cross_ent')

  def call(self, inputs, **kwargs):
    if tf.executing_eagerly():
      return self.classifier(inputs, **kwargs)
    else:
      # TODO(basilm): Find a way around neeed for variable_scope - using
      #   tf.enable_resource_variables() doesn't seem to work.
      with tf.compat.v1.variable_scope('scope', use_resource=True):
        return self.classifier(inputs, **kwargs)


class DenseClassifier(tf.python.keras.Model):
  """Feedforward neural network with MCSoftmaxDense output layer."""

  def __init__(self, num_classes, logit_noise=tfp.distributions.Normal,
               temperature=1.0, train_mc_samples=1000, test_mc_samples=1000,
               compute_pred_variance=False):
    """Creates an instance of DenseClassifier.

    A feedforward network which computes the predictive and log predictive
    distribution.

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
        variance.

    Returns:
      DenseClassifier instance.
    """
    super(DenseClassifier, self).__init__()

    self.hidden_layer = tf.python.keras.layers.Dense(16)
    self.output_layer = ed.layers.MCSoftmaxDense(
        num_classes=num_classes, logit_noise=logit_noise,
        temperature=temperature, train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance)

  def call(self, inputs, training=True, seed=None):
    """Computes the forward pass through the feedforward neural network.

    Args:
      inputs: `Tensor`. Input tensor.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      A tuple of `Tensors` (probs, log_probs, predictive_variance).
    """
    hidden_x = self.hidden_layer(inputs)
    return self.output_layer(hidden_x, training=training, seed=seed)


class DenseFAClassifier(tf.python.keras.Model):
  """Feedforward neural network with MCSoftmaxDenseFA output layer."""

  def __init__(self, num_classes, num_factors,
               temperature=1.0, parameter_efficient=False,
               train_mc_samples=1000, test_mc_samples=1000,
               compute_pred_variance=False):
    """Creates an instance of DenseFAClassifier.

    A feedforward network which computes the predictive and log predictive
    distribution.

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use for factor analysis approx.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance.

    Returns:
      DenseFAClassifier instance.
    """
    super(DenseFAClassifier, self).__init__()

    self.hidden_layer = tf.python.keras.layers.Dense(16)
    self.output_layer = ed.layers.MCSoftmaxDenseFA(
        num_classes=num_classes, num_factors=num_factors,
        temperature=temperature, parameter_efficient=parameter_efficient,
        train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance)

  def call(self, inputs, training=True, seed=None):
    """Computes the forward pass through the feedforward neural network.

    Args:
      inputs: `Tensor`. Input tensor.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      A tuple of `Tensors` (probs, log_probs, predictive_variance).
    """
    hidden_x = self.hidden_layer(inputs)
    return self.output_layer(hidden_x, training=training, seed=seed)


class SigmoidDenseFAClassifier(tf.python.keras.Model):
  """Feedforward neural network with MCSigmoidDenseFA output layer."""

  def __init__(self, num_classes, num_factors,
               temperature=1.0, parameter_efficient=False,
               train_mc_samples=1000, test_mc_samples=1000,
               compute_pred_variance=False):
    """Creates an instance of SigmoidDenseFAClassifier.

    A feedforward network which computes the predictive and log predictive
    distribution.

    Args:
      num_classes: Integer. Number of classes for classification task.
      num_factors: Integer. Number of factors to use for factor analysis approx.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      parameter_efficient: Boolean. Whether to use the parameter efficient
        version of the method.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance.

    Returns:
      SigmoidDenseFAClassifier instance.
    """
    super(SigmoidDenseFAClassifier, self).__init__()

    self.hidden_layer = tf.python.keras.layers.Dense(16)
    self.output_layer = ed.layers.MCSigmoidDenseFA(
        1 if num_classes == 2 else num_classes, num_factors=num_factors,
        temperature=temperature, parameter_efficient=parameter_efficient,
        train_mc_samples=train_mc_samples,
        test_mc_samples=test_mc_samples,
        compute_pred_variance=compute_pred_variance)

  def call(self, inputs, training=True, seed=None):
    """Computes the forward pass through the feedforward neural network.

    Args:
      inputs: `Tensor`. Input tensor.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      A tuple of `Tensors` (probs, log_probs, predictive_variance).
    """
    hidden_x = self.hidden_layer(inputs)
    return self.output_layer(hidden_x, training=training, seed=seed)


class ExactSigmoidDenseClassifier(tf.python.keras.Model):
  """Feedforward neural network with ExactSigmoidDense output layer."""

  def __init__(self, num_classes, logit_noise):
    """Creates an instance of ExactSigmoidDenseClassifier.

    A feedforward network which computes the predictive and log predictive
    distribution.

    Args:
      num_classes: Integer. Number of classes for classification task.
      logit_noise: tfp.distributions instance. Must be either
        tfp.distributions.Normal or tfp.distributions.Logistic.

    Returns:
      ExactSigmoidDenseClassifier instance.
    """
    super(ExactSigmoidDenseClassifier, self).__init__()

    self.hidden_layer = tf.python.keras.layers.Dense(16)
    self.output_layer = ed.layers.ExactSigmoidDense(
        1 if num_classes == 2 else num_classes, logit_noise=logit_noise)

  def call(self, inputs, training=True, seed=None):
    """Computes the forward pass through the feedforward neural network.

    Args:
      inputs: `Tensor`. Input tensor.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      A tuple of `Tensors` (probs, log_probs, predictive_variance).
    """
    hidden_x = self.hidden_layer(inputs)
    return self.output_layer(hidden_x, training=training)


class EnsembleClassifier(tf.python.keras.Model):
  """Feedforward neural network with Ensemble output layer."""

  def __init__(self, num_classes, averaging, ensemble_weighting=(0.8, 0.2)):
    """Creates an instance of EnsembleClassifier.

    A feedforward network which computes the predictive and log predictive
    distribution.

    Args:
      num_classes: Integer. Number of classes for classification task.
      averaging: String `ensemble_cross_ent` or `gibbs_cross_ent`. For
        `ensemble_cross_ent`: loss = - log (sum_i  weighting[i] * p_i)
        i.e. ensemble members are trained in the knowledge they will be
        ensembled. For `gibbs_cross_ent`:
        loss = - sum_i weighting[i] * log (p_i), this can help promote
        diversity.
      ensemble_weighting: Tuple of len(layers) representing a probability
        distribution over layers.

    Returns:
      EnsembleClassifier instance.
    """
    super(EnsembleClassifier, self).__init__()

    self.hidden_layer = tf.python.keras.layers.Dense(16)
    if num_classes == 2:
      layer_1 = ed.layers.MCSigmoidDenseFA(1)
      layer_2 = ed.layers.ExactSigmoidDense(1)
    else:
      layer_1 = ed.layers.MCSoftmaxDense(num_classes=num_classes)
      layer_2 = ed.layers.MCSoftmaxDenseFA(num_classes=num_classes,
                                           num_factors=num_classes//2)

    self.output_layer = ed.layers.EnsembleHeteroscedasticOutputs(
        num_classes, (layer_1, layer_2),
        ensemble_weighting=ensemble_weighting, averaging=averaging)

  def call(self, inputs, training=True):
    """Computes the forward pass through the feedforward neural network.

    Args:
      inputs: `Tensor`. Input tensor.
      training: Boolean. Whether we are training or not.

    Returns:
      A tuple of `Tensors` (probs, log_probs, predictive_variance).
    """
    hidden_x = self.hidden_layer(inputs)
    return self.output_layer(hidden_x, training=training)


class HeteroscedasticLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    if not tf.executing_eagerly():
      tf.compat.v1.enable_resource_variables()
    super().setUp()

  # Helpers for HeteroscedasticLibTest.
  def create_dataset(self, num_classes):
    x = np.asarray([[1.0, 2.0], [0.5, 1.5], [0.2, 0.15], [-0.3, 0.0]])

    y = np.asarray([[i % num_classes] for i in range(4)])

    return tf.convert_to_tensor(x), tf.convert_to_tensor(y)

  @test_cases()
  def test_layer_construction(self, logit_noise, num_classes, model_type):
    if model_type == 'MCSoftmaxDense':
      output_layer = ed.layers.MCSoftmaxDense(num_classes=num_classes,
                                              logit_noise=logit_noise)
      self.assertIsNotNone(output_layer)

  @test_cases()
  def test_model_construction(self, logit_noise, num_classes, model_type):
    if model_type == 'MCSoftmaxDense':
      classifier = DenseClassifier(num_classes, logit_noise)
    elif model_type == 'EnsembleEnsembleCE':
      classifier = EnsembleClassifier(num_classes, 'ensemble_cross_ent')
    elif model_type == 'EnsembleGibbsCE':
      classifier = EnsembleClassifier(num_classes, 'gibbs_cross_ent')
    else:
      return

    self.assertIsNotNone(classifier)

  def test_ensemble_weighting(self):
    classifier = EnsembleClassifier(
        2, 'ensemble_cross_ent', ensemble_weighting=(0.5, 0.5))
    self.assertIsNotNone(classifier)

    classifier = EnsembleClassifier(
        2, 'ensemble_cross_ent', ensemble_weighting=(0.8, 0.2))
    self.assertIsNotNone(classifier)

    with self.assertRaises(ValueError):
      classifier = EnsembleClassifier(
          2, 'ensemble_cross_ent', ensemble_weighting=(0.4, 0.5))

    with self.assertRaises(ValueError):
      classifier = EnsembleClassifier(
          2, 'ensemble_cross_ent', ensemble_weighting=(1.5, -0.5))

  @test_cases()
  def test_model_outputs(self, logit_noise, num_classes, model_type):
    x, _ = self.create_dataset(num_classes)

    classifier = Classifier(model_type, num_classes, logit_noise)

    res = classifier(x)
    probs = res[2]
    log_probs = res[1]

    self.assertIsNotNone(probs)
    self.assertIsNotNone(log_probs)

    self.initialise()
    if num_classes == 2 or 'Sigmoid' in model_type:
      for prob in self.evaluate(probs).flatten():
        self.assertAlmostEqual(prob + (1.0 - prob), 1.0, 2)
    else:
      total_probs = tf.reduce_sum(probs, axis=-1)
      for prob in self.evaluate(total_probs).flatten():
        self.assertAlmostEqual(prob, 1.0, 2)

    res = classifier(x, training=False)
    probs = res[2]
    log_probs = res[1]

    self.assertIsNotNone(probs)
    self.assertIsNotNone(log_probs)

    if num_classes == 2 or 'Sigmoid' in model_type:
      for prob in self.evaluate(probs).flatten():
        self.assertAlmostEqual(prob + (1.0 - prob), 1.0, 2)
    else:
      total_probs = tf.reduce_sum(probs, axis=-1)
      for prob in self.evaluate(total_probs).flatten():
        self.assertAlmostEqual(prob, 1.0, 2)

  @test_cases()
  def test_train_step(self, logit_noise, num_classes, model_type):
    x, y = self.create_dataset(num_classes)

    classifier = Classifier(model_type, num_classes, logit_noise)

    if num_classes == 2:
      loss_fn = tf.python.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
      loss_fn = tf.python.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    if tf.executing_eagerly():
      optimizer = tf.python.keras.optimizers.Adam()
      def train_step(inputs, labels, model):
        """Defines a single training step: Update weights based on one batch."""
        with tf.GradientTape() as tape:
          log_preds = model(inputs)[1]
          loss_value = loss_fn(labels, log_preds)

        grads = tape.gradient(loss_value, model.trainable_weights)
        grads, _ = tf.clip_by_global_norm(grads, 2.5)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

      loss_value = train_step(x, y, classifier).numpy()
    else:
      optimizer = tf.compat.v1.train.AdamOptimizer()
      log_preds = classifier(x)[1]
      loss_value = loss_fn(y, log_preds)
      train_op = optimizer.minimize(loss_value)
      self.initialise()
      loss_value, _ = self.evaluate([loss_value, train_op])

    self.assertGreater(loss_value, 0)

  @test_cases()
  def test_predictive_variance(self, logit_noise, num_classes, model_type):
    if model_type == 'Exact' or model_type.startswith('Ensemble'):
      return
    x, _ = self.create_dataset(num_classes)

    classifier = Classifier(model_type, num_classes, logit_noise,
                            compute_pred_variance=True)

    pred_variance = classifier(x)[3]

    self.assertIsNotNone(pred_variance)
    self.initialise()
    pred_variance = self.evaluate(pred_variance)
    for per_class_variance in pred_variance.flatten():
      self.assertGreater(per_class_variance, 0)

  def initialise(self):
    if not tf.executing_eagerly():
      self.evaluate([tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer()])

if __name__ == '__main__':
  tf.test.main()
