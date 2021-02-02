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

"""Gaussian process layers."""

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import initializers
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow as tf
import tensorflow_probability as tfp


class Zeros(object):
  """Function returning zeros tensor of same shape excluding the last dim."""

  def __call__(self, inputs):
    return tf.zeros(tf.shape(inputs)[:-1], inputs.dtype)

  def get_config(self):
    return {}


class ExponentiatedQuadratic(object):
  """Exponentiated quadratic kernel."""

  def __init__(self, variance, lengthscale):
    self.variance = variance
    self.lengthscale = lengthscale

  def __call__(self, x1, x2):
    """Computes exponentiated quadratic over all pairs of inputs.

    Args:
      x1: Tensor of shape [batch_x1, ...]. Slices along the batch axis denote an
        individual input to be passed to the kernel. It is computed pairwise
        with each input sliced from x2.
      x2: Tensor of shape [batch_x2, ...]. Slices along the batch axis denote an
        individual input passed to the kernel function. It is computed pairwise
        with each input sliced from x1.

    Returns:
      Tensor of shape [batch_x1, batch_x2].
    """
    size = tf.convert_to_tensor(x1).shape.ndims
    if size > 2:
      raise NotImplementedError('Multiple feature dimensions is not yet '
                                'supported.')
    x1 = x1 / self.lengthscale
    x2 = x2 / self.lengthscale
    x1_squared = tf.reduce_sum(tf.square(x1), list(range(1, len(x1.shape))))
    x2_squared = tf.reduce_sum(tf.square(x2), list(range(1, len(x2.shape))))
    square = (x1_squared[:, tf.newaxis] +
              x2_squared[tf.newaxis, :] -
              2 * tf.matmul(x1, x2, transpose_b=True))
    return self.variance * tf.exp(-square / 2)

  def get_config(self):
    return {'variance': self.variance, 'lengthscale': self.lengthscale}


class LinearKernel(object):
  """Linear kernel, optionally on top of a feature extractor (e.g., encoder)."""

  def __init__(self, variance, bias, encoder=tf.identity):
    self.variance = variance
    self.bias = bias
    self.encoder = encoder

  def __call__(self, x1, x2):
    """Computes scaled dot product of over all pairs of encoded inputs.

    Args:
      x1: Tensor of shape [batch_x1] + encoder domain. Slices along the batch
        axis denote an individual input to be passed to the kernel. It is
        computed pairwise with each input sliced from x2.
      x2: Tensor of shape [batch_x2] + encoder domain. Slices along the batch
        axis denote an individual input to be passed to the kernel. It is
        computed pairwise with each input sliced from x1.

    Returns:
      Tensor of shape [batch_x1, batch_x2].
    """
    encoded_x1 = self.encoder(x1)
    encoded_x2 = self.encoder(x2)
    dot_product = tf.matmul(encoded_x1, encoded_x2, transpose_b=True)
    return self.variance * dot_product + self.bias

  def get_config(self):
    return {
        'variance': self.variance,
        'bias': self.bias,
        'encoder': tf.python.keras.utils.serialize_keras_object(self.encoder),
    }


class GaussianProcess(tf.python.keras.layers.Layer):
  r"""Gaussian process layer.

  The layer represents a distribution over functions, where a
  stochastic forward pass appears as

  ```none
  f ~ GP(f | conditional_inputs, conditional_outputs; mean_fn, covariance_fn)
  outputs = f(inputs)
  ```

  The optional arguments `conditional_inputs` and `conditional_outputs`
  capture data that the GP "memorizes", i.e., it forms a posterior predictive
  distribution. If left unspecified, the GP posits a prior predictive.

  Given a call to `inputs`, an equivalent formulation in terms of function
  outputs is

  ```none
  outputs ~ \prod_{unit=1}^{units} MultivariateNormal(output[:, unit] |
      mean = mean_fn(inputs) + Knm Kmm^{-1} (conditional_outputs[:, unit]-mean),
      covariance = Knn - Knm Kmm^{-1} Kmn)
  ```

  where Knm is the covariance function evaluated between all `inputs` and
  `conditional_inputs`; Knn is between all `inputs`; Kmm is between all
  `conditional_inputs`; and mean is the mean function evaluated on
  `conditional_inputs`. The multivariate normal is correlated across input
  dimensions and is independent across output dimensions.
  """

  def __init__(
      self,
      units,
      mean_fn=Zeros(),
      covariance_fn=ExponentiatedQuadratic(variance=1., lengthscale=1.),
      conditional_inputs=None,
      conditional_outputs=None,
      **kwargs):
    """Constructs layer.

    Args:
      units: integer, dimensionality of layer.
      mean_fn: Mean function, a callable taking an inputs Tensor of shape
        [batch, ...] and returning a Tensor of shape [batch].
      covariance_fn: Covariance function, a callable taking two input Tensors
        of shape [batch_x1, ...] and [batch_x2, ...] respectively, and returning
        a positive semi-definite matrix of shape [batch_x1, batch_x2].
      conditional_inputs: Tensor of shape [batch, ...], where batch must be the
        same as conditional_outputs', and ellipses must match layer inputs.
      conditional_outputs: Tensor of shape [batch, units], where batch must be
        the same as conditional_inputs' and units is the layer's units size.
      **kwargs: kwargs passed to parent class.
    """
    super(GaussianProcess, self).__init__(**kwargs)
    self.units = int(units)
    self.mean_fn = mean_fn
    self.covariance_fn = covariance_fn
    self.conditional_inputs = conditional_inputs
    self.conditional_outputs = conditional_outputs

    self.supports_masking = True
    self.input_spec = tf.python.keras.layers.InputSpec(min_ndim=2)

  def build(self, input_shape=None):
    # Don't track trainable variables such as in the kernel. The user should
    # refer to any via, e.g., self.covariance_fn or the user environment.
    self.built = True

  def call(self, inputs):
    if self.conditional_inputs is None and self.conditional_outputs is None:
      covariance_matrix = self.covariance_fn(inputs, inputs)
      # Tile locations so output has shape [units, batch_size]. Covariance will
      # broadcast to [units, batch_size, batch_size], and we perform
      # shape manipulations to get a random variable over [batch_size, units].
      loc = self.mean_fn(inputs)
      loc = tf.tile(loc[tf.newaxis], [self.units] + [1] * len(loc.shape))
    else:
      knn = self.covariance_fn(inputs, inputs)
      knm = self.covariance_fn(inputs, self.conditional_inputs)
      kmm = self.covariance_fn(self.conditional_inputs, self.conditional_inputs)
      kmm = tf.linalg.set_diag(
          kmm, tf.linalg.diag_part(kmm) + tf.python.keras.backend.epsilon())
      kmm_tril = tf.linalg.cholesky(kmm)
      kmm_tril_operator = tf.linalg.LinearOperatorLowerTriangular(kmm_tril)
      knm_operator = tf.linalg.LinearOperatorFullMatrix(knm)

      # TODO(trandustin): Vectorize linear algebra for multiple outputs. For
      # now, we do each separately and stack to obtain a locations Tensor of
      # shape [units, batch_size].
      loc = []
      for conditional_outputs_unit in tf.unstack(self.conditional_outputs,
                                                 axis=-1):
        center = conditional_outputs_unit - self.mean_fn(
            self.conditional_inputs)
        loc_unit = knm_operator.matvec(
            kmm_tril_operator.solvevec(kmm_tril_operator.solvevec(center),
                                       adjoint=True))
        loc.append(loc_unit)
      loc = tf.stack(loc) + self.mean_fn(inputs)[tf.newaxis]

      covariance_matrix = knn
      covariance_matrix -= knm_operator.matmul(
          kmm_tril_operator.solve(
              kmm_tril_operator.solve(knm, adjoint_arg=True), adjoint=True))

    covariance_matrix = tf.linalg.set_diag(
        covariance_matrix,
        tf.linalg.diag_part(covariance_matrix) + tf.python.keras.backend.epsilon())

    # Form a multivariate normal random variable with batch_shape units and
    # event_shape batch_size. Then make it be independent across the units
    # dimension. Then transpose its dimensions so it is [batch_size, units].
    random_variable = (
        generated_random_variables.MultivariateNormalFullCovariance(
            loc=loc, covariance_matrix=covariance_matrix))
    random_variable = generated_random_variables.Independent(
        random_variable.distribution, reinterpreted_batch_ndims=1)
    bijector = tfp.bijectors.Inline(
        forward_fn=lambda x: tf.transpose(x, perm=[1, 0]),
        inverse_fn=lambda y: tf.transpose(y, perm=[1, 0]),
        forward_event_shape_fn=lambda input_shape: input_shape[::-1],
        forward_event_shape_tensor_fn=lambda input_shape: input_shape[::-1],
        inverse_log_det_jacobian_fn=lambda y: tf.cast(0, y.dtype),
        forward_min_event_ndims=2)
    random_variable = generated_random_variables.TransformedDistribution(
        random_variable.distribution, bijector=bijector)
    return random_variable

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    input_dim = input_shape[-1]
    if input_dim is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = {
        'units': self.units,
        'mean_fn': tf.python.keras.utils.serialize_keras_object(self.mean_fn),
        'covariance_fn': tf.python.keras.utils.serialize_keras_object(
            self.covariance_fn),
        'conditional_inputs': None,  # don't serialize as it can be large
        'conditional_outputs': None,  # don't serialize as it can be large
    }
    base_config = super(GaussianProcess, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@utils.add_weight
class SparseGaussianProcess(GaussianProcess):
  r"""Gaussian process layer with inducing input and output variables.

  The layer represents a distribution over functions, where a
  stochastic forward pass appears as

  ```none
  f ~ GP(f | inducing_inputs, inducing_outputs; mean_fn, covariance_fn)
  outputs = f(inputs)
  ```

  The arguments `inducing_inputs` and `inducing_outputs`
  capture data that the GP "memorizes", i.e., it forms a posterior predictive
  distribution. Typically in a variational inference scheme (and by default),
  the inducing outputs are normally distributed with learnable location and
  scale parameters, and the inducing inputs are learnable parameters.

  Given a call to `inputs` with these defaults, an equivalent formulation in
  terms of function outputs is

  ```none
  inducing_outputs ~ Normal(inducing_outputs | mean, stddev)
  outputs ~ \prod_{unit=1}^{units} MultivariateNormal(output[:, unit] |
      mean = mean_fn(inputs) + Knm Kmm^{-1} (inducing_outputs[:, unit]-mean),
      covariance = Knn - Knm Kmm^{-1} Kmn)
  ```

  where Knm is the covariance function evaluated between all `inputs` and
  `inducing_inputs`; Knn is between all `inputs`; Kmm is between all
  `inducing_inputs`; and mean is the mean function evaluated on
  `inducing_inputs`. The multivariate normal is correlated across input
  dimensions and is independent across output dimensions.

  #### Examples

  We demonstrate a three-layer deep GP with variational inference (Salimbeni and
  Deisenroth, 2017; Damianou and Lawrence, 2013). The code snippet mirrors
  Figure 5 of Bayesian Layers. We apply it for regression given batches of
  spatial inputs and vector-valued outputs. We flatten inputs to use the
  default squared exponential kernel; this naturally extends to pass in a
  more sophisticated kernel function.

  ```python
  from tensor2tensor.layers import bayes

  batch_size = 256
  dataset_size = 10000
  features, labels = load_spatial_data(batch_size)

  model = tf.python.keras.Sequential([
    tf.python.keras.layers.Flatten(),
    layers.SparseGaussianProcess(256, num_inducing=512),
    layers.SparseGaussianProcess(256, num_inducing=512),
    layers.SparseGaussianProcess(10, num_inducing=512),
  ])

  # Run training loop.
  num_steps = 1000
  for _ in range(num_steps):
    with tf.GradientTape() as tape:
      predictions = model(features)
      nll = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
      kl = sum(model.losses) / dataset_size
      loss = nll + kl
    gradients = tape.gradient(loss, model.variables)  # use any optimizer here
  ```
  """

  def __init__(
      self,
      units,
      num_inducing,
      mean_fn=Zeros(),
      covariance_fn=ExponentiatedQuadratic(variance=1., lengthscale=1.),
      inducing_inputs_initializer='random_normal',
      inducing_outputs_initializer='trainable_normal',
      inducing_inputs_regularizer=None,
      inducing_outputs_regularizer='normal_kl_divergence',
      inducing_inputs_constraint=None,
      inducing_outputs_constraint=None,
      **kwargs):
    """Constructs layer.

    Args:
      units: integer, dimensionality of layer.
      num_inducing: integer, number of inducing points for the approximation.
      mean_fn: Mean function, a callable taking an inputs Tensor of shape
        [batch, ...] and returning a Tensor of shape [batch].
      covariance_fn: Covariance function, a callable taking two input Tensors
        of shape [batch_x1, ...] and [batch_x2, ...] respectively, and returning
        a positive semi-definite matrix of shape [batch_x1, batch_x2].
      inducing_inputs_initializer: Initializer for the inducing inputs.
      inducing_outputs_initializer: Initializer for the inducing outputs.
      inducing_inputs_regularizer: Regularizer function applied to the inducing
        inputs.
      inducing_outputs_regularizer: Regularizer function applied to the inducing
        outputs.
      inducing_inputs_constraint: Constraint function applied to the inducing
        inputs.
      inducing_outputs_constraint: Constraint function applied to the inducing
        outputs.
      **kwargs: kwargs passed to parent class.
    """
    super(SparseGaussianProcess, self).__init__(
        units=units,
        mean_fn=mean_fn,
        covariance_fn=covariance_fn,
        conditional_inputs=None,
        conditional_outputs=None,
        **kwargs)
    self.num_inducing = num_inducing
    self.inducing_inputs_initializer = initializers.get(
        inducing_inputs_initializer)
    self.inducing_outputs_initializer = initializers.get(
        inducing_outputs_initializer)
    self.inducing_inputs_regularizer = regularizers.get(
        inducing_inputs_regularizer)
    self.inducing_outputs_regularizer = regularizers.get(
        inducing_outputs_regularizer)
    self.inducing_inputs_constraint = constraints.get(
        inducing_inputs_constraint)
    self.inducing_outputs_constraint = constraints.get(
        inducing_outputs_constraint)

  def build(self, input_shape=None):
    input_shape = tf.TensorShape(input_shape)
    input_dim = input_shape[-1]
    self.conditional_inputs = self.add_weight(
        shape=(self.num_inducing, input_dim),
        name='inducing_inputs',
        initializer=self.inducing_inputs_initializer,
        regularizer=self.inducing_inputs_regularizer,
        constraint=self.inducing_inputs_constraint)
    self.conditional_outputs = self.add_weight(
        shape=(self.num_inducing, self.units),
        name='inducing_outputs',
        initializer=self.inducing_outputs_initializer,
        regularizer=self.inducing_outputs_regularizer,
        constraint=self.inducing_outputs_constraint)
    super(SparseGaussianProcess, self).build(input_shape)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.inducing_inputs_initializer, tf.python.keras.layers.Layer):
      self.conditional_inputs = self.inducing_inputs_initializer(
          self.conditional_inputs.shape, self.dtype)
    if isinstance(self.inducing_outputs_initializer, tf.python.keras.layers.Layer):
      self.conditional_outputs = self.inducing_outputs_initializer(
          self.conditional_outputs.shape, self.dtype)

  def call(self, inputs):
    self.call_weights()
    return super(SparseGaussianProcess, self).call(inputs)
