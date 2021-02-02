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

# Lint as: python3
"""Regression model for generalized neural processes.
"""

import edward2 as ed
from experimental.attentive_uncertainty import attention  # local file import
from experimental.attentive_uncertainty import layers  # local file import
from experimental.attentive_uncertainty import utils  # local file import

import tensorflow.compat.v1 as tf

eps = tf.python.keras.backend.epsilon()


class Regressor(tf.python.keras.Model):
  r"""Generalized neural process regressor.

  A generalized neural process (GNP) expresses the following generative process

  ```
  z ~ p(z | global_latent_layer(C))
  zi ~ p(zi | local_latent_layer(z, xi, C))
  yi ~ p(yi | decoder(z, zi, xi, C))
  ```

  Maximizing the marginal likelihood is intractable and SNPs maximize the
  evidence lower bound obtained via the following variational distributions

  ```
  z ~ q(z | global_latent_layer(T))
  zi ~ q(zi | local_latent_layer(z, xi, T))
  ```

  Note that the global_latent_net and local_latent_net parameters are shared.
  Different instantiations of GNP differ in the particular forms of
  conditioning they use; in particular, what ancestors to condition on and how
  to condition (via attention or not).
  """

  def __init__(self,
               input_dim,
               output_dim,
               x_encoder_net_sizes=None,
               x_y_encoder_net_sizes=None,
               heteroskedastic_net_sizes=None,
               global_latent_net_sizes=None,
               local_latent_net_sizes=None,
               decoder_net_sizes=None,
               att_type='multihead',
               att_heads=8,
               model_type='fully_connected',
               activation=tf.nn.relu,
               output_activation=None,
               model_path=None,
               data_uncertainty=True,
               beta=1.,
               temperature=1.):
    """Initializes the generalized neural process regressor.

    D below denotes:
    - Context dataset C during decoding phase
    - Target dataset T during encoding phase

    Args:
      input_dim: (int) Dimensionality of covariates x.
      output_dim: (int) Dimensionality of labels y.
      x_encoder_net_sizes: (list of ints) Hidden layer sizes for network
        featurizing x.
      x_y_encoder_net_sizes: (list of ints) Hidden layer sizes for network
        featurizing D.
      heteroskedastic_net_sizes: (list of ints) Hidden layer sizes for network
      that maps x to heteroskedastic variance.
      global_latent_net_sizes: (list of ints) Hidden layer sizes for network
        that maps D to mean and variance of predictive p(z | D).
      local_latent_net_sizes: (list of ints) Hidden layer sizes for network
        that maps xi, z, D to mean and variance of predictive p(zi | z, xi, D).
      decoder_net_sizes: (list of ints) Hidden layer sizes for network that maps
        xi, z, zi, D to mean and variance of predictive p(yi | z, zi, xi, D).
      att_type: (string) Attention type for freeform attention.
      att_heads: (int) Number of heads in case att_type='multihead'.
      model_type: (string) One of 'fully_connected', 'cnp', 'acnp', 'acns',
        'np', 'anp'.
      activation: (callable) Non-linearity used for all neural networks.
      output_activation: (callable) Non-linearity for predictive mean.
      model_path: (string) File path for best early-stopped model.
      data_uncertainty: (boolean) True if data uncertainty is explicit.
      beta: (float) Scaling factor for global kl loss.
      temperature: (float) Inverse scaling factor for temperature.

    Raises:
      ValueError: If model_type is unrecognized.
    """
    if (model_type not in
        ['np', 'anp', 'acns', 'fully_connected', 'cnp', 'acnp']):
      raise ValueError('Unrecognized model type: %s'% model_type)

    super(Regressor, self).__init__()
    self._input_dim = input_dim
    self._output_dim = output_dim
    self.model_type = model_type
    self._output_activation = output_activation
    self._data_uncertainty = data_uncertainty
    self.beta = tf.constant(beta)
    self.temperature = temperature

    self._global_latent_layer = None
    self._local_latent_layer = None
    self._decoder_layer = None
    self._dataset_encoding_layer = None
    self._x_encoder = None
    self._heteroskedastic_net = None
    self._homoskedastic_net = None

    contains_global = ['np', 'anp', 'acns', 'fully_connected']
    contains_local = ['acns', 'fully_connected']

    x_dim = input_dim
    if x_encoder_net_sizes is not None:
      self._x_encoder = utils.mlp_block(
          input_dim,
          x_encoder_net_sizes,
          activation)
      x_dim = x_encoder_net_sizes[-1]

    x_y_net = None
    self_dataset_attention = None
    if x_y_encoder_net_sizes is not None:
      x_y_net = utils.mlp_block(
          x_dim + output_dim,
          x_y_encoder_net_sizes,
          activation)
      dataset_encoding_dim = x_y_encoder_net_sizes[-1]
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
        att_type=att_type, num_heads=att_heads, scale=self.temperature)
    self._cross_dataset_attention.build([x_dim, dataset_encoding_dim])

    if model_type in contains_global:
      global_latent_net = utils.mlp_block(
          dataset_encoding_dim,
          global_latent_net_sizes,
          activation)
      self._global_latent_layer = layers.GlobalLatentLayer(global_latent_net)
      global_latent_dim = global_latent_net_sizes[-1]//2

    if model_type in contains_local:
      local_input_dim = global_latent_dim + dataset_encoding_dim
      local_latent_net = utils.mlp_block(
          local_input_dim,
          local_latent_net_sizes,
          activation)
      self._local_latent_layer = layers.LocalLatentLayer(local_latent_net)
      local_latent_dim = local_latent_net_sizes[-1]//2

      separate_prior_net = (model_type != 'fully_connected')
      if separate_prior_net:
        local_latent_net = utils.mlp_block(
            global_latent_dim,
            local_latent_net_sizes,
            activation)
        self._prior_local_latent_layer = layers.LocalLatentLayer(
            local_latent_net)
      else:
        self._prior_local_latent_layer = self._local_latent_layer

    if decoder_net_sizes is not None:
      decoder_input_dim = x_dim
      if model_type == 'cnp' or model_type == 'acnp':  # depend on C
        decoder_input_dim += dataset_encoding_dim
      elif model_type == 'np':  # depend on z
        decoder_input_dim += global_latent_dim
      elif model_type == 'anp':  # depend on z, C
        decoder_input_dim += dataset_encoding_dim + global_latent_dim
      elif model_type == 'acns':
        decoder_input_dim += dataset_encoding_dim + local_latent_dim
      elif model_type == 'fully_connected':
        decoder_input_dim += (dataset_encoding_dim + global_latent_dim
                              + local_latent_dim)
      decoder_net = utils.mlp_block(
          decoder_input_dim,
          decoder_net_sizes,
          activation)
      self._decoder_layer = layers.DecoderLayer(
          decoder_net,
          model_type,
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
      global_z_kl = self.beta * global_z_posterior.distribution.kl_divergence(
          global_z_prior.distribution)
    else:
      global_z_kl = tf.constant(0., shape=(1, 1))
    self.add_loss(lambda: global_z_kl)
    cross_attentive_encodings = None
    if self.model_type not in ['cnp', 'np']:
      cross_attentive_encodings = self._cross_dataset_attention(
          target_x, context_x, context_x_y_encodings)

    posterior_x_y_encodings = None
    prior_x_y_encodings = None
    if self.model_type == 'fully_connected':
      prior_x_y_encodings = cross_attentive_encodings
      posterior_x_y_encodings = self._cross_dataset_attention(
          target_x, target_x, target_x_y_encodings)
    else:
      posterior_x_y_encodings = target_x_y_encodings
    if self.model_type == 'cnp':
      cross_attentive_encodings = tf.tile(
          avg_context_dataset_encodings,
          [1, tf.shape(target_x)[1], 1])

    local_z_prior = None
    local_z_posterior = None
    num_targets = tf.shape(target_x)[1]
    if self._local_latent_layer is not None:
      local_z_prior = self._prior_local_latent_layer(
          global_z_prior,
          num_targets,
          prior_x_y_encodings)
      if target_y is None:
        local_z_posterior = local_z_prior
      else:
        local_z_posterior = self._local_latent_layer(
            global_z_posterior,
            num_targets,
            posterior_x_y_encodings)
      local_z_kl = local_z_posterior.distribution.kl_divergence(
          local_z_prior.distribution)
    else:
      local_z_kl = tf.constant(0., shape=(1, 1, 1))
    self.add_loss(lambda: local_z_kl)

    predictive = self._decoder_layer(
        target_x,
        cross_attentive_encodings,
        local_z_posterior,
        global_z_posterior)

    posterior_predictive_mean = predictive.distribution.mean()
    posterior_predictive_std = tf.sqrt(
        tf.square(predictive.distribution.stddev()) + data_var + eps)
    posterior_predictive = ed.Normal(loc=posterior_predictive_mean,
                                     scale=posterior_predictive_std)

    return posterior_predictive
