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

"""Regression model for structured neural processes.
"""

import edward2 as ed
from experimental.attentive_uncertainty import attention  # local file import
from experimental.attentive_uncertainty import layers  # local file import
from experimental.attentive_uncertainty import utils  # local file import

import tensorflow.compat.v1 as tf

eps = tf.python.keras.backend.epsilon()


class Regressor(tf.python.keras.Model):
  r"""Structured neural process regressor.

  A structured neural process (SNP) expresses the following generative process

  ```
  z ~ p(z | global_latent_layer(C))
  zi ~ p(zi | local_latent_layer(z, xi, C))
  yi ~ p(yi | zi, heteroskedastic_net(xi)/homoskedastic_net(None))
  ```

  Maximizing the marginal likelihood is intractable and SNPs maximize the
  evidence lower bound obtained via the following variational distributions

  ```
  z ~ q(z | global_latent_layer(T))
  zi ~ q(zi | local_latent_layer(z, xi, T)
  ```

  Note that the global_latent_net and local_latent_net parameters are used
  for inference as well for parameter efficiency.
  """

  def __init__(self,
               input_dim,
               output_dim=1,
               x_encoder_sizes=None,
               x_y_encoder_sizes=None,
               heteroskedastic_net_sizes=None,
               global_latent_net_sizes=None,
               local_latent_net_sizes=None,
               att_type='multihead',
               att_heads=8,
               uncertainty_type='attentive_freeform',
               mean_att_type=attention.laplace_attention,
               scale_att_type_1=attention.squared_exponential_attention,
               scale_att_type_2=attention.squared_exponential_attention,
               activation=tf.nn.relu,
               output_activation=None,
               model_path=None,
               data_uncertainty=True,
               local_variational=True):
    """Initializes the structured neural process regressor.

    D below denotes:
    - Context dataset C during decoding phase
    - Target dataset T during encoding phase

    Args:
      input_dim: (int) Dimensionality of covariates x.
      output_dim: (int) Dimensionality of labels y.
      x_encoder_sizes: (list of ints) Hidden layer sizes for featurizing x.
      x_y_encoder_sizes: (list of ints) Hidden layer sizes for featurizing C/D.
      heteroskedastic_net_sizes: (list of ints) Hidden layer sizes for network
      that maps x to heteroskedastic variance.
      global_latent_net_sizes: (list of ints) Hidden layer sizes for network
        that maps D to mean and variance of predictive p(z | D).
      local_latent_net_sizes: (list of ints) Hidden layer sizes for network
        that maps xi, z, D to mean and variance of predictive p(z_i | z, xi, D).
      att_type: (string) Attention type for freeform attention.
      att_heads: (int) Number of heads in case att_type='multihead'.
      uncertainty_type: (string) One of 'attentive_gp', 'attentive_freeform'.
        Default is 'attentive_freeform' which does not impose structure on
        posterior mean, std.
      mean_att_type: (call) Attention for mean of predictive p(zi | z, x, D).
      scale_att_type_1: (call) Attention for std of predictive p(zi | z, x, D).
      scale_att_type_2: (call) Attention for std of predictive p(zi | z, x, D).
      activation: (callable) Non-linearity used for all neural networks.
      output_activation: (callable) Non-linearity for predictive mean.
      model_path: (string) File path for best early-stopped model.
      data_uncertainty: (boolean) True if data uncertainty is explicit.
      local_variational: (boolean) True if VI performed on local latents.
    """
    super(Regressor, self).__init__()
    self._input_dim = input_dim
    self._output_dim = output_dim
    self._uncertainty_type = uncertainty_type
    self._output_activation = output_activation
    self._data_uncertainty = data_uncertainty
    self.local_variational = local_variational
    self._global_latent_layer = None
    self._local_latent_layer = None
    self._dataset_encoding_layer = None
    self._x_encoder = None
    self._heteroskedastic_net = None
    self._homoskedastic_net = None

    x_dim = input_dim
    if x_encoder_sizes is not None:
      self._x_encoder = utils.mlp_block(
          input_dim,
          x_encoder_sizes,
          activation)
      x_dim = x_encoder_sizes[-1]

    x_y_net = None
    self_dataset_attention = None
    if x_y_encoder_sizes is not None:
      x_y_net = utils.mlp_block(
          x_dim + output_dim,
          x_y_encoder_sizes,
          activation)
      dataset_encoding_dim = x_y_encoder_sizes[-1]
    else:
      # Use self-attention.
      dataset_encoding_dim = x_dim + output_dim
      self_dataset_attention = attention.AttentionLayer(
          att_type=att_type, num_heads=att_heads)
      self_dataset_attention.build([x_dim, x_dim])

    self._dataset_encoding_layer = layers.DatasetEncodingLayer(
        x_y_net,
        self_dataset_attention)
    self._cross_dataset_attention = attention.AttentionLayer(
        att_type=att_type, num_heads=att_heads)
    self._cross_dataset_attention.build([x_dim, dataset_encoding_dim])

    local_latent_dim = x_dim
    if global_latent_net_sizes is not None:
      global_latent_net = utils.mlp_block(
          dataset_encoding_dim,
          global_latent_net_sizes,
          activation)
      self._global_latent_layer = layers.GlobalLatentLayer(global_latent_net)
      local_latent_dim += global_latent_net_sizes[-1]//2

    if local_latent_net_sizes is not None:
      # Freeform uncertainty directly attends to dataset encoding.
      if uncertainty_type == 'attentive_freeform':
        local_latent_dim += dataset_encoding_dim

      local_latent_net = utils.mlp_block(
          local_latent_dim,
          local_latent_net_sizes,
          activation)
      self._local_latent_layer = layers.SNPLocalLatentLayer(
          local_latent_net,
          uncertainty_type,
          mean_att_type,
          scale_att_type_1,
          scale_att_type_2,
          output_activation)

    if data_uncertainty:
      if heteroskedastic_net_sizes is not None:
        self._heteroskedastic_net = utils.mlp_block(
            x_dim,
            heteroskedastic_net_sizes,
            activation)
      else:
        self._homoskedastic_net = layers.DataNoise()
        self._homoskedastic_net.build(None)

    if model_path:
      self.load_weights(model_path)

  def call(self, context_x, context_y, target_x, target_y=None):
    if self._x_encoder is not None:
      context_x = self._x_encoder(context_x)
      target_x = self._x_encoder(target_x)

    if self._data_uncertainty:
      if self._heteroskedastic_net is None:
        data_var = tf.nn.softplus(self._homoskedastic_net(None))
      else:
        data_var = tf.nn.softplus(self._heteroskedastic_net(target_x))
    else:
      data_var = 0.

    context_x_y_encodings = self._dataset_encoding_layer(context_x, context_y)
    if target_y is None:
      target_x_y_encodings = context_x_y_encodings
    else:
      target_x_y_encodings = self._dataset_encoding_layer(target_x, target_y)
    avg_context_dataset_encodings = tf.reduce_mean(
        context_x_y_encodings, axis=1, keepdims=True)
    avg_target_dataset_encodings = tf.reduce_mean(
        target_x_y_encodings, axis=1, keepdims=True)

    global_z_prior = None
    global_z_posterior = None
    if self._global_latent_layer is not None:
      global_z_prior = self._global_latent_layer(avg_context_dataset_encodings)
      global_z_posterior = self._global_latent_layer(
          avg_target_dataset_encodings)
      global_z_kl = global_z_posterior.distribution.kl_divergence(
          global_z_prior.distribution)
      self.add_loss(lambda: global_z_kl)

    self_attentive_encodings = None
    cross_attentive_encodings = None
    if self._uncertainty_type == 'attentive_freeform':
      cross_attentive_encodings = self._cross_dataset_attention(
          target_x, context_x, context_x_y_encodings)
      if target_y is None:
        self_attentive_encodings = cross_attentive_encodings
      else:
        self_attentive_encodings = self._cross_dataset_attention(
            target_x, target_x, target_x_y_encodings)

    # TODO(adityagrover): Variational inference for hyperparameters.
    lengthscale_1 = 1.
    lengthscale_2 = 1.
    local_z_prior = self._local_latent_layer(
        target_x,
        context_x,
        context_y,
        global_z_prior,
        cross_attentive_encodings,
        lengthscale_1=lengthscale_1,
        lengthscale_2=lengthscale_2)

    if self.local_variational:
      if target_y is None:
        local_z_posterior = local_z_prior
      else:
        local_z_posterior = self._local_latent_layer(
            target_x,
            target_x,
            target_y,
            global_z_posterior,
            self_attentive_encodings,
            lengthscale_1=lengthscale_1,
            lengthscale_2=lengthscale_2)

      local_z_kl = local_z_posterior.distribution.kl_divergence(
          local_z_prior.distribution)
      self.add_loss(lambda: local_z_kl)

      posterior_predictive_mean = local_z_posterior.distribution.mean()
      posterior_predictive_std = tf.sqrt(
          tf.square(local_z_posterior.distribution.stddev()) + data_var + eps)
      posterior_predictive = ed.Normal(loc=posterior_predictive_mean,
                                       scale=posterior_predictive_std)
      output_predictive = posterior_predictive
    else:
      prior_predictive_mean = local_z_prior.distribution.mean()
      prior_predictive_std = tf.sqrt(
          tf.square(local_z_prior.distribution.stddev()) + data_var + eps)
      prior_predictive = ed.Normal(loc=prior_predictive_mean,
                                   scale=prior_predictive_std)
      # With no variational inference, local_z_kl term is zero.
      self.add_loss(lambda: tf.constant(0., shape=(1, 1, 1)))
      output_predictive = prior_predictive

    return output_predictive
