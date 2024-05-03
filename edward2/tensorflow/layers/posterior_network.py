# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Library of methods to compute Posterior Network predictions."""

from typing import Optional, Union, Iterable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class PosteriorNetworkLayer(tf.keras.layers.Layer):
  """Output layer for a Posterior Network model."""

  def __init__(self,
               num_classes: int,
               flow_type: str = 'maf',
               flow_depth: int = 8,
               flow_width: Optional[int] = None,
               class_counts: Optional[Iterable[int]] = None,
               name: str = 'PosteriorNetworkLayer'):
    """Makes a Posterior Network output layer.

    Args:
      num_classes: (int) Number of output classes.
      flow_type: Type of the normalizing flow to be used; has to be one
                      of 'maf', 'radial', or 'affine'.
      flow_depth: Number of latent flows to stack into a deep flow.
      flow_width: Width of the hidden layers inside the MAF flows.
      class_counts: List of ints counting the training examples per class.
      name: Name of the layer.
    """
    super().__init__(name=name)
    self.num_classes = num_classes
    self.flow_type = flow_type
    self.flow_depth = flow_depth
    self.flow_width = flow_width
    if class_counts is None:
      class_counts = tf.ones(num_classes)
    self.class_counts = class_counts

  def build(self, input_shape):
    """Builds the layer based on the passed input shape."""
    with tf.name_scope(self.name):
      # Using the PyTorch default hyperparameters.
      self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                                           momentum=0.9)
      self.latent_dim = input_shape[-1]
      if self.flow_width is None:
        self.flow_width = 2 * self.latent_dim
      self.flows = []
      for _ in range(self.num_classes):
        flow = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.latent_dim, dtype=tf.float32),
                scale_diag=tf.ones(self.latent_dim, dtype=tf.float32)),
            bijector=_make_deep_flow(self.flow_type,
                                     self.flow_depth,
                                     self.flow_width,
                                     self.latent_dim))
        self.flows.append(flow)

  def call(self, inputs, training=True, return_probs=False):
    latents = self.batch_norm(inputs)
    log_ps = [self.flows[i].log_prob(latents) for i in range(self.num_classes)]
    log_ps_stacked = tf.stack(log_ps, axis=-1)
    alphas = 1. + (self.class_counts * tf.exp(log_ps_stacked))
    probs, _ = tf.linalg.normalize(alphas, ord=1, axis=-1)
    if return_probs:
      return probs
    else:
      return alphas

  def get_config(self):
    config = super().get_config()
    config.update({
        'num_classes': self.num_classes,
        'flow_type': self.flow_type,
        'flow_depth': self.flow_depth,
        'flow_width': self.flow_width,
        'class_counts': self.class_counts
    })
    return config


class ReversedRadialFlow(tfp.bijectors.Bijector):
  """A radial flow, but backwards.

  The original radial flow (Rezende & Mohamed 2016) is not invertible for any
  arbitrary values of alpha and beta and relies on constraining their values.
  Link to paper: https://arxiv.org/abs/1505.05770

  Since in our particular application of posterior networks, we do not
  intend to sample from the model, we do not need the _forward() function.
  Instead, we only use it as a density estimator, so we need the _log_prob()
  function, which depends on the _inverse() pass and respective log det of the
  Jacobian. Hence, these functions are implemented the way that would
  normally be the forward pass through the flow, such that we do not need to
  constrain alpha and beta.
  """

  def __init__(self,
               dim: int,
               x0: Optional[Union[tf.Tensor, np.ndarray]] = None,
               alpha_prime_init: Optional[float] = None,
               beta_prime_init: Optional[float] = None,
               validate_args: bool = False,
               name: str = 'radial'):
    """Builds a reversed radial flow model.

    Args:
      dim: (int) Dimensionality of the input and output space.
      x0: (array,Tensor) Reference point in the input space.
      alpha_prime_init: (float) Alpha' parameter for the radial flow.
      beta_prime_init: (float) Beta' parameter for the radial flow.
      validate_args: (bool) Flag to validate the arguments.
      name: (str) Name for the model.
    """
    super().__init__(
        validate_args=validate_args,
        inverse_min_event_ndims=1,
        name=name)
    with tf.name_scope(name) as name:
      self._name = name
      if x0 is None:
        x0 = tf.zeros(dim)
      else:
        x0 = tf.convert_to_tensor(x0)
        if x0.shape[-1] != tf.TensorShape(dim):
          raise ValueError(f'Variable x0={x0} needs to have shape [{dim}]. '
                           f'Found shape {x0.shape}')
      self.x0 = tf.Variable(x0, name='x0', dtype=tf.float32, trainable=True)
      # if alpha' and beta' are not defined, we sample them the same way
      # that Pyro does (https://docs.pyro.ai/en/latest/_modules/pyro/distributions/transforms/radial.html)  # pylint: disable=line-too-long
      if alpha_prime_init is None:
        alpha_prime_init = tf.random.uniform([], -1 / np.sqrt(dim),
                                             1 / np.sqrt(dim))
      if beta_prime_init is None:
        beta_prime_init = tf.random.uniform([], -1/np.sqrt(dim),
                                            1/np.sqrt(dim))
      self.alpha_prime = tf.Variable(
          alpha_prime_init, name='alpha', dtype=tf.float32, trainable=True)
      self.beta_prime = tf.Variable(
          beta_prime_init, name='beta', dtype=tf.float32, trainable=True)
      self.dim = dim

  def _forward(self, z):
    """The normal radial flow is not invertible, so this is not defined."""
    raise NotImplementedError("Forward shouldn't be called!")

  def _get_alpha_beta(self):
    """Regularize alpha' and beta' to get alpha and beta."""
    alpha = tf.nn.softplus(self.alpha_prime)
    beta = -alpha + tf.nn.softplus(self.beta_prime)
    return alpha, beta

  def _inverse(self, x):
    """The forward pass of the original radial flow, following the paper.

    This is the first line of Eq. (14) in https://arxiv.org/abs/1505.05770.

    Args:
      x: Input to the flow.
    Returns:
      The transformed output tensor of the flow.
    """
    alpha, beta = self._get_alpha_beta()
    diff = x - self.x0
    r = tf.linalg.norm(diff, axis=-1, keepdims=True)
    h = 1. / (alpha + r)
    beta_h = beta * h
    return x + beta_h * diff

  def _inverse_log_det_jacobian(self, x):
    """Computes the log det Jacobian, as per the paper.

    This is the second line of Eq. (14) in https://arxiv.org/abs/1505.05770.
    Args:
      x: Input to the flow.
    Returns:
      The log determinant of the Jacobian of the flow.
    """
    alpha, beta = self._get_alpha_beta()
    diff = x - self.x0
    r = tf.linalg.norm(diff, axis=-1, keepdims=True)
    h = 1. / (alpha + r)
    h_prime = -(h ** 2)
    beta_h = beta * h
    log_det_jacobian = tf.reduce_sum(
        (self.dim - 1) * tf.math.log1p(beta_h)
        + tf.math.log1p(beta_h + beta * h_prime * r), axis=-1)
    return log_det_jacobian


def _make_deep_flow(flow_type, flow_depth, flow_width, dim):
  """Builds a deep flow of the specified type."""
  if flow_type not in ['maf', 'radial', 'affine']:
    raise ValueError(f'Flow type {flow_type} is not maf, radial, or affine.')
  if flow_type == 'maf':
    return _make_maf_flow(flow_depth, flow_width)
  elif flow_type == 'radial':
    return _make_radial_flow(dim, flow_depth)
  elif flow_type == 'affine':
    return _make_affine_flow(dim, flow_depth)


def _make_maf_flow(flow_depth, flow_width):
  """Builds a deep stack of masked autoregressive flows."""
  # If not otherwise specified, make the hidden layers of the flow twice
  # as wide as the latent dimension, to make them expressive enough to
  # parameterize a shift and scale for each dimension.
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  # Build the deep MAF flow.
  # Each layer outputs two params per dimension, for shift and scale.
  bijectors.append(
      tfp.bijectors.MaskedAutoregressiveFlow(
          tfp.bijectors.AutoregressiveNetwork(
              params=2, hidden_units=[flow_width]*flow_depth,
              activation='relu')))
  # For numerical stability of training, we need these batch norms.
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))


def _make_radial_flow(dim, flow_depth):
  """Builds a deep stack of radial flows."""
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  for _ in range(flow_depth):
    bijectors.append(ReversedRadialFlow(dim))
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))


def _make_affine_flow(dim, flow_depth):
  """Builds a deep stack of affine flows."""
  bijectors = []
  bijectors.append(tfp.bijectors.BatchNormalization())
  for _ in range(flow_depth):
    bijectors.append(
        tfp.bijectors.Shift(tf.Variable(tf.zeros(dim), trainable=True)))
    bijectors.append(
        tfp.bijectors.ScaleMatvecDiag(
            tf.Variable(tf.ones(dim), trainable=True)))
  bijectors.append(tfp.bijectors.BatchNormalization())
  return tfp.bijectors.Chain(list(reversed(bijectors)))
