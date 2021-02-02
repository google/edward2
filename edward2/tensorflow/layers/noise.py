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

"""Bayesian noise layers."""

from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import random_variable

import tensorflow as tf
import tensorflow_probability as tfp


class NCPNormalPerturb(tf.python.keras.layers.Layer):
  """Noise contrastive prior for continuous inputs (Hafner et al., 2018).

  The layer doubles the inputs' batch size and adds a random normal perturbation
  to the concatenated second batch. This acts an input prior to be used in
  combination with an output prior. The output prior reduces the second batch
  (reverting to the inputs' original shape) and computes a regularizer that
  matches the second batch towards some output (e.g., uniform distribution).
  This layer implementation is inspired by the Aboleth library.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = ed.layers.NCPNormalPerturb()(inputs)  # double input batch
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  means = ed.layers.DenseVariationalDropout(1, activation=None)(x)  # get mean
  means = ed.layers.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.python.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.python.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.python.keras.Model(inputs=inputs, outputs=outputs)

  # Run training loop.
  num_steps = 1000
  for _ in range(num_steps):
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
      loss += model.losses[0] / dataset_size  # KL regularizer for output layer
      loss += model.losses[-1]
    gradients = tape.gradient(loss, model.variables)  # use any optimizer here
  ```

  The network applies `ed.layers.NCPNormalPerturb()` to double the input batch
  size and add Gaussian noise to the second half; then feedforward layers; then
  `ed.layers.DenseVariational` to be Bayesian about the output density's mean;
  then `ed.layers.NCPNormalOutput` centered at the labels to revert to the batch
  size and compute a loss on the second half; then parameterize the output
  density's standard deviations; then compute the total loss function as the sum
  of the model's negative log-likelihood, KL divergence for the Bayesian mean
  layer, and NCP loss.
  """

  def __init__(self, mean=0., stddev=1., seed=None, **kwargs):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    super(NCPNormalPerturb, self).__init__(**kwargs)

  def call(self, inputs):
    noise = tf.random.normal(tf.shape(inputs),
                             mean=self.mean,
                             stddev=self.stddev,
                             dtype=inputs.dtype,
                             seed=self.seed)
    perturbed_inputs = inputs + noise
    return tf.concat([inputs, perturbed_inputs], 0)


class NCPCategoricalPerturb(tf.python.keras.layers.Layer):
  """Noise contrastive prior for discrete inputs (Hafner et al., 2018).

  The layer doubles the inputs' batch size and randomly flips categories
  for the concatenated second batch (all features must be integer-valued). This
  acts an input prior to be used in combination with an output prior. The output
  prior reduces the second batch (reverting to the inputs' original shape) and
  computes a regularizer that matches the second batch towards some output
  (e.g., uniform distribution). This layer implementation is inspired by the
  Aboleth library.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = ed.layers.NCPCategoricalPerturb(10)(inputs)  # double input batch
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  means = ed.layers.DenseVariationalDropout(1, activation=None)(x)  # get mean
  means = ed.layers.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.python.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.python.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.python.keras.Model(inputs=inputs, outputs=outputs)

  # Run training loop.
  num_steps = 1000
  for _ in range(num_steps):
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
      loss += model.losses[0] / dataset_size  # KL regularizer for output layer
      loss += model.losses[-1]
    gradients = tape.gradient(loss, model.variables)  # use any optimizer here
  ```

  The network applies `ed.layers.NCPCategoricalPerturb()` to double the input
  batch size and flip categories for the second half; then feedforward layers;
  then `ed.layers.DenseVariational` to be Bayesian about the output density's
  mean; then `ed.layers.NCPNormalOutput` centered at the labels to revert to the
  batch size and compute a loss on the second half; then parameterize the output
  density's standard deviations; then compute the total loss function as the sum
  of the model's negative log-likelihood, KL divergence for the Bayesian mean
  layer, and NCP loss.
  """

  def __init__(self, input_dim, probs=0.1, **kwargs):
    """Creates layer.

    Args:
      input_dim: int > 0. Size of the category, i.e. maximum integer index + 1.
      probs: Probability that a category is randomly flipped.
      **kwargs: kwargs to parent class.
    """
    self.input_dim = input_dim
    self.probs = probs
    super(NCPCategoricalPerturb, self).__init__(**kwargs)

  def call(self, inputs):
    mask = tf.cast(tf.random.uniform(tf.shape(inputs)) <= self.probs,
                   inputs.dtype)
    flips = tf.random.uniform(
        tf.shape(inputs), minval=0, maxval=self.input_dim, dtype=inputs.dtype)
    flipped_inputs = mask * flips + (1 - mask) * inputs
    return tf.concat([inputs, flipped_inputs], 0)


class NCPNormalOutput(tf.python.keras.layers.Layer):
  """Noise contrastive prior for continuous outputs (Hafner et al., 2018).

  The layer returns the first half of the inputs' batch. It computes a KL
  regularizer as a side-effect, which matches the inputs' second half towards a
  normal distribution (the output prior), and averaged over the number of inputs
  in the second half. This layer is typically in combination with an input prior
  which doubles the batch. This layer implementation is inspired by the Aboleth
  library.

  The layer computes the exact KL divergence from a normal distribution to
  the input RandomVariable. It is an unbiased estimate if the input
  RandomVariable has random parameters. If the input is a Tensor, then it
  assumes its density is `ed.Normal(input, 1.)`, i.e., mean squared error loss.

  #### Examples

  Below implements neural network regression with heteroskedastic noise,
  noise contrastive priors, and being Bayesian only at the mean's output layer.

  ```python
  batch_size, dataset_size = 128, 1000
  features, labels = get_some_dataset()

  inputs = keras.Input(shape=(25,))
  x = ed.layers.NCPNormalPerturb()(inputs)  # double input batch
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  x = tf.python.keras.layers.Dense(64, activation='relu')(x)
  means = ed.layers.DenseVariationalDropout(1, activation=None)(x)  # get mean
  means = ed.layers.NCPNormalOutput(labels)(means)  # halve input batch
  stddevs = tf.python.keras.layers.Dense(1, activation='softplus')(x[:batch_size])
  outputs = tf.python.keras.layers.Lambda(lambda x: ed.Normal(x[0], x[1]))([means,
                                                                     stddevs])
  model = tf.python.keras.Model(inputs=inputs, outputs=outputs)

  # Run training loop.
  num_steps = 1000
  for _ in range(num_steps):
    with tf.GradientTape() as tape:
      predictions = model(features)
      loss = tf.reduce_mean(predictions.distribution.log_prob(labels))
      loss += model.losses[0] / dataset_size  # KL regularizer for output layer
      loss += model.losses[-1]
    gradients = tape.gradient(loss, model.variables)  # use any optimizer here
  ```

  The network applies `ed.layers.NCPNormalPerturb()` to double the input batch
  size and add Gaussian noise to the second half; then feedforward layers; then
  `ed.layers.DenseVariational` to be Bayesian about the output density's mean;
  then `ed.layers.NCPNormalOutput` centered at the labels to revert to the batch
  size and compute a loss on the second half; then parameterize the output
  density's standard deviations; then compute the total loss function as the sum
  of the model's negative log-likelihood, KL divergence for the Bayesian mean
  layer, and NCP loss.
  """

  def __init__(self, mean=0., stddev=1., **kwargs):
    self.mean = mean
    self.stddev = stddev
    super(NCPNormalOutput, self).__init__(**kwargs)

  def call(self, inputs):
    if not isinstance(inputs, random_variable.RandomVariable):
      # Default to a unit normal, i.e., derived from mean squared error loss.
      inputs = generated_random_variables.Normal(loc=inputs, scale=1.)
    batch_size = tf.shape(inputs)[0] // 2
    # TODO(trandustin): Depend on github's ed2 for indexing RVs. This is a hack.
    # _, _ = inputs[:batch_size], inputs[batch_size:]
    original_inputs = random_variable.RandomVariable(
        inputs.distribution[:batch_size],
        value=inputs.value[:batch_size])
    perturbed_inputs = random_variable.RandomVariable(
        inputs.distribution[batch_size:],
        value=inputs.value[batch_size:])
    loss = tf.reduce_sum(
        tfp.distributions.Normal(self.mean, self.stddev).kl_divergence(
            perturbed_inputs.distribution))
    loss /= tf.cast(batch_size, dtype=tf.float32)
    self.add_loss(loss)
    return original_inputs
