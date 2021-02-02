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

"""Neural process."""

from edward2.tensorflow import generated_random_variables

import tensorflow as tf
import tensorflow_probability as tfp


def batch_mlp(inputs, hidden_sizes):
  """Apply MLP to the final axis of a 3D tensor.

  Args:
    inputs: input Tensor of shape [batch_size, n, d_in].
    hidden_sizes: An iterable containing the hidden layer sizes of the MLP.

  Returns:
    Tensor of shape [batch_size, n, d_out] where d_out = output_sizes[-1].
  """
  inputs = tf.convert_to_tensor(inputs)
  batch_size, _, filter_size = inputs.shape.as_list()
  hidden = tf.reshape(inputs, (-1, filter_size))

  for size in hidden_sizes[:-1]:
    hidden = tf.python.keras.layers.Dense(size, activation=tf.nn.relu)(hidden)

  output = tf.python.keras.layers.Dense(hidden_sizes[-1], activation=None)(hidden)
  output = tf.reshape(output, (batch_size, -1, hidden_sizes[-1]))
  return output


# TODO(adityagrover): Reimplement using preexisting attention routines in T2T
def uniform_attention(q, v):
  """Computes uniform attention. Equivalent to neural process.

  Args:
    q: queries. Tensor of shape [batch_size, m, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  total_points = tf.shape(q)[1]
  rep = tf.reduce_mean(v, axis=1, keepdims=True)  # [batch_size, 1, d_v]
  rep = tf.tile(rep, [1, total_points, 1])
  return rep


def laplace_attention(q, k, v, scale, normalise):
  """Computes laplace exponential attention.

  Args:
    q: queries. Tensor of shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    scale: float that scales the L1 distance.
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  k = tf.expand_dims(k, axis=1)  # [batch_size, 1, n, d_k]
  q = tf.expand_dims(q, axis=2)  # [batch_size, m, 1, d_k]
  unnorm_weights = - tf.abs((k - q) / scale)  # [batch_size, m, n, d_k]
  unnorm_weights = tf.reduce_sum(unnorm_weights, axis=-1)  # [batch_size, m, n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = lambda x: 1 + tf.tanh(x)
  weights = weight_fn(unnorm_weights)  # [batch_size, m, n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [batch_size, m, d_v]
  return rep


def dot_product_attention(q, k, v, normalise):
  """Computes dot product attention.

  Args:
    q: queries. Tensor of  shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    normalise: Boolean that determines whether weights sum to 1.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  d_k = tf.shape(q)[-1]
  scale = tf.sqrt(tf.cast(d_k, tf.float32))
  unnorm_weights = tf.einsum('bjk,bik->bij', k, q) / scale  # [batch_size,m,n]
  if normalise:
    weight_fn = tf.nn.softmax
  else:
    weight_fn = tf.sigmoid
  weights = weight_fn(unnorm_weights)  # [batch_size,m,n]
  rep = tf.einsum('bik,bkj->bij', weights, v)  # [batch_size,m,d_v]
  return rep


def multihead_attention(q, k, v, num_heads=8):
  """Computes multi-head attention.

  Args:
    q: queries. Tensor of  shape [batch_size, m, d_k].
    k: keys. Tensor of shape [batch_size, n, d_k].
    v: values. Tensor of shape [batch_size, n, d_v].
    num_heads: number of heads. Should divide d_v.

  Returns:
    Tensor of shape [batch_size, m, d_v].
  """
  d_k = q.shape.as_list()[-1]
  d_v = v.shape.as_list()[-1]
  head_size = int(d_v / num_heads)
  key_initializer = tf.python.keras.initializers.RandomNormal(stddev=d_k**-0.5)
  value_initializer = tf.python.keras.initializers.RandomNormal(stddev=d_v**-0.5)
  rep = tf.constant(0.0)
  for h in range(num_heads):
    o = dot_product_attention(
        tf.python.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wq%d' % h, use_bias=False, padding='VALID')(q),
        tf.python.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wk%d' % h, use_bias=False, padding='VALID')(k),
        tf.python.keras.layers.Conv1D(
            head_size, 1, kernel_initializer=key_initializer,
            name='wv%d' % h, use_bias=False, padding='VALID')(v),
        normalise=True)
    rep += tf.python.keras.layers.Conv1D(d_v, 1, kernel_initializer=value_initializer,
                                  name='wo%d' % h, use_bias=False,
                                  padding='VALID')(o)
  return rep


# TODO(adityagrover): Implement via T2T.
class Attention(object):
  """The Attention module."""

  def __init__(self, rep, output_sizes, att_type, scale=1., normalise=True,
               num_heads=8):
    """Creates a attention module.

    Takes in context inputs, target inputs and
    representations of each context input/output pair
    to output an aggregated representation of the context data.

    Args:
      rep: transformation to apply to contexts before computing attention.
          One of: ['identity', 'mlp'].
      output_sizes: list of number of hidden units per layer of mlp.
          Used only if rep == 'mlp'.
      att_type: type of attention. One of the following:
          ['uniform', 'laplace', 'dot_product', 'multihead']
      scale: scale of attention.
      normalise: Boolean determining whether to:
          1. apply softmax to weights so they sum to 1 across context pts or
          2. apply custom transformation to have weights in [0, 1].
      num_heads: number of heads for multihead.
    """
    self._rep = rep
    self._output_sizes = output_sizes
    self._type = att_type
    self._scale = scale
    self._normalise = normalise
    if self._type == 'multihead':
      self._num_heads = num_heads

  def __call__(self, x1, x2, r):
    """Applies attention to create aggregated representation of r.

    Args:
      x1: Tensor of shape [B ,n1, d_x].
      x2: Tensor of shape [batch_size, n2, d_x].
      r: Tensor of shape [batch_size, n1, d].

    Returns:
      Tensor of shape [batch_size, n2, d]

    Raises:
      NameError: The argument for rep/type was invalid.
    """
    if self._rep == 'identity':
      k, q = (x1, x2)
    elif self._rep == 'mlp':
      k = batch_mlp(x1, self._output_sizes)
      q = batch_mlp(x2, self._output_sizes)
    else:
      raise NameError("'rep' not among ['identity', 'mlp']")

    if self._type == 'uniform':
      rep = uniform_attention(q, r)
    elif self._type == 'laplace':
      rep = laplace_attention(q, k, r, self._scale, self._normalise)
    elif self._type == 'dot_product':
      rep = dot_product_attention(q, k, r, self._normalise)
    elif self._type == 'multihead':
      rep = multihead_attention(q, k, r, self._num_heads)
    else:
      raise NameError(("'att_type' not among ['uniform', 'laplace', "
                       "'dot_product', 'multihead']"))

    return rep


# TODO(adityagrover): Make the encoder and decoder configurable.
class NeuralProcess(tf.python.keras.Model):
  """Attentive Neural Process (Kim et al., 2019; Garnelo et al., 2018)."""

  def __init__(self,
               latent_encoder_sizes,
               num_latents,
               decoder_sizes,
               use_deterministic_path=True,
               deterministic_encoder_sizes=None,
               attention_wrapper=None):
    """Initializes the Neural Process model.

    Args:
      latent_encoder_sizes: (list of ints) Hidden layer sizes for latent
          encoder.
      num_latents: (int) Dimensionality of global latent variable.
      decoder_sizes: (list of ints) Hidden layer sizes for decoder
      use_deterministic_path: (bool) Uses deterministic encoder as well if True.
      deterministic_encoder_sizes: (list of ints) Hidden layer sizes for
          deterministic encoder.
      attention_wrapper: Instance of Attention class to apply for
          determinitic encoder embedding.
    """
    super(NeuralProcess, self).__init__()
    self._num_latents = num_latents
    self._latent_encoder_sizes = latent_encoder_sizes
    self._deterministic_encoder_sizes = deterministic_encoder_sizes
    self._decoder_sizes = decoder_sizes
    self._use_deterministic_path = use_deterministic_path
    self._attention = attention_wrapper

  def latent_encoder(self, x, y):
    """Encodes the inputs into one representation.

    Args:
      x: Tensor of shape [batch_size, observations, d_x]. For the prior, these
         are context x-values. For the posterior, these are target x-values.
      y: Tensor of shape [batch_size, observations, d_y]. For the prior, these
         are context y-values. For the posterior, these are target y-values.

    Returns:
      A normal distribution over tensors of shape [batch_size, num_latents].
    """
    encoder_input = tf.concat([x, y], axis=-1)
    per_example_embedding = batch_mlp(
        encoder_input, self._latent_encoder_sizes)
    dataset_embedding = tf.reduce_mean(per_example_embedding, axis=1)
    hidden = tf.python.keras.layers.Dense(
        (self._latent_encoder_sizes[-1] + self._num_latents)//2,
        activation=tf.nn.relu)(dataset_embedding)
    loc = tf.python.keras.layers.Dense(self._num_latents, activation=None)(hidden)
    untransformed_scale = tf.python.keras.layers.Dense(self._num_latents,
                                                activation=None)(hidden)
    # Constraint scale following Garnelo et al. (2018).
    scale_diag = 0.1 + 0.9 * tf.sigmoid(untransformed_scale)
    return generated_random_variables.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag)

  def deterministic_encoder(self, context_x, context_y, target_x):
    """Encodes the inputs into one representation.

    Args:
      context_x: Tensor of shape [batch_size, observations, d_x].
        Observed x-values.
      context_y: Tensor of shape [batch_size, observations, d_y].
        Observed y-values.
      target_x: Tensor of shape [batch_size, target_observations, d_x].
        Target x-values.

    Returns:
      Encodings. Tensor of shape [batch_size, target_observations, d].
    """
    encoder_input = tf.concat([context_x, context_y], axis=-1)
    per_example_embedding = batch_mlp(encoder_input,
                                      self._deterministic_encoder_sizes)
    per_target_embedding = self._attention(context_x,
                                           target_x,
                                           per_example_embedding)
    return per_target_embedding

  def decoder(self, representation, target_x):
    """Decodes the individual targets.

    Args:
      representation: The representation of the context for target predictions.
          Tensor of shape [batch_size, target_observations, ?].
      target_x: The x locations for the target query.
          Tensor of shape [batch_size, target_observations, d_x].

    Returns:
      dist: A multivariate Gaussian over the target points. A distribution over
          tensors of shape [batch_size, target_observations, d_y].
    """
    decoder_input = tf.concat([representation, target_x], axis=-1)
    hidden = batch_mlp(decoder_input, self._decoder_sizes)
    loc, untransformed_scale = tf.split(hidden, 2, axis=-1)
    scale_diag = 0.1 + 0.9 * tf.nn.softplus(untransformed_scale)
    return tfp.distributions.MultivariateNormalDiag(loc=loc,
                                                    scale_diag=scale_diag)

  def __call__(self, query, target_y=None):
    """Returns the predicted mean and variance at the target points.

    Args:
      query: Nested tuple containing ((context_x, context_y), target_x) where:
              context_x is Tensor of shape [batch_size, num_contexts, d_x].
                  Contains the x values of the context points.
              context_y is Tensor of shape [batch_size, num_contexts, d_y].
                  Contains the y values of the context points.
              target_x is Tensor of shape [batch_size, num_targets, d_x].
                  Contains the x values of the target points.
      target_y: The ground truth y values of the target y.
          Tensor of shape [batch_size, num_targets, d_y].

    Returns:
      predictive_dist: Predictive posterior distribution over the predicted y.
    """

    (context_x, context_y), target_x = query
    num_targets = tf.shape(target_x)[1]
    prior = self.latent_encoder(context_x, context_y)

    # For training, when target_y is available, use targets for latent encoder.
    # Note that targets contain contexts by design.
    # For testing, when target_y unavailable, use contexts for latent encoder.
    if target_y is None:
      latent_rep = prior
    else:
      posterior = self.latent_encoder(target_x, target_y)
      latent_rep = posterior
    latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                         [1, num_targets, 1])
    if self._use_deterministic_path:
      deterministic_rep = self.deterministic_encoder(context_x,
                                                     context_y,
                                                     target_x)
      representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
    else:
      representation = latent_rep

    predictive_dist = self.decoder(representation, target_x)

    if target_y is not None:
      kl = tf.expand_dims(
          posterior.distribution.kl_divergence(prior.distribution),
          -1)
      self.add_loss(lambda: kl)

    return predictive_dist

  call = __call__
