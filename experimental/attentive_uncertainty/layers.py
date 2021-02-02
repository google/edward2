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

"""Utility layers for structured and generalized neural processes.
"""

import edward2 as ed
import tensorflow.compat.v1 as tf

eps = tf.python.keras.backend.epsilon()


class DataNoise(tf.python.keras.layers.Layer):
  """Creates a variable for modeling homoskedastic noise."""

  def build(self, input_shape=None):
    self.untransformed_data_var = self.add_weight(
        shape=input_shape,
        initializer=tf.random_normal_initializer(),
        dtype=tf.float32,
        name='data_noise')

  def call(self, inputs):
    return self.untransformed_data_var


class DatasetEncodingLayer(tf.python.keras.layers.Layer):
  """Encodes a dataset of (x, y) pairs into embeddings via a shared network."""

  def __init__(self,
               net,
               dataset_attention):
    super(DatasetEncodingLayer, self).__init__()
    self._net = net
    self._dataset_attention = dataset_attention

  def call(self, x, y):
    dataset = tf.concat([x, y], axis=-1)
    if self._net is not None:
      x_y_encodings = self._net(dataset)
    else:
      # Use self-attention.
      x_y_encodings = self._dataset_attention(dataset, dataset, dataset)

    return x_y_encodings


class GlobalLatentLayer(tf.python.keras.layers.Layer):
  """Maps embedded (x, y) points to a single stochastic embedding."""

  def __init__(self, net):
    super(GlobalLatentLayer, self).__init__()
    self._net = net

  def call(self, avg_dataset_encodings):
    logits = self._net(avg_dataset_encodings)
    mean, untransformed_std = tf.split(logits, 2, axis=-1)
    std = tf.nn.softplus(untransformed_std) + eps
    return ed.Normal(loc=mean, scale=std)


class LocalLatentLayer(tf.python.keras.layers.Layer):
  """Maps conditioning inputs to a per-point stochastic embedding."""

  def __init__(self, net):
    super(LocalLatentLayer, self).__init__()
    self._net = net

  def call(self,
           sampled_global_latents,
           num_targets,
           local_x_y_encodings):
    inputs = tf.tile(sampled_global_latents, [1, num_targets, 1])
    if local_x_y_encodings is not None:
      inputs = tf.concat([inputs, local_x_y_encodings], axis=-1)
    logits = self._net(inputs)
    mean, untransformed_std = tf.split(logits, 2, axis=-1)
    std = tf.nn.softplus(untransformed_std)
    return ed.Normal(loc=mean, scale=std)


class DecoderLayer(tf.python.keras.layers.Layer):
  """Maps conditioning inputs to a per-point predictive distribution."""

  def __init__(self,
               net,
               model_type,
               output_activation=None):
    super(DecoderLayer, self).__init__()
    self._net = net
    self._model_type = model_type
    self._output_activation = output_activation

  def call(self,
           unlabelled_x,
           attentive_encodings,
           sampled_local_latents,
           sampled_global_latents):
    inputs = unlabelled_x
    if self._model_type in ['cnp', 'acnp', 'anp', 'acns', 'fully_connected']:
      inputs = tf.concat([inputs, attentive_encodings], axis=-1)
    if self._model_type in ['acns', 'fully_connected']:
      inputs = tf.concat([inputs, sampled_local_latents], axis=-1)
    if self._model_type in ['np', 'anp', 'fully_connected']:
      tiled_global_latents = tf.tile(
          sampled_global_latents,
          [1, tf.shape(unlabelled_x)[1], 1])
      inputs = tf.concat([inputs, tiled_global_latents], axis=-1)
    logits = self._net(inputs)
    mean, untransformed_std = tf.split(logits, 2, axis=-1)
    if self._output_activation is not None:
      mean = self._output_activation(mean)
    std = tf.nn.softplus(untransformed_std)
    return ed.Normal(loc=mean, scale=std)


class SNPLocalLatentLayer(tf.python.keras.layers.Layer):
  """Maps each datapoint (and global conditioning) to stochastic embedding."""

  def __init__(self,
               net,
               uncertainty_type,
               mean_att_type,
               scale_att_type_1,
               scale_att_type_2,
               output_activation=None):
    super(SNPLocalLatentLayer, self).__init__()
    self._net = net
    self._uncertainty_type = uncertainty_type
    self._attention_mean = mean_att_type
    self._attention_scale_1 = scale_att_type_1
    self._attention_scale_2 = scale_att_type_2
    self._output_activation = output_activation

  def call(self,
           unlabelled_x,
           labelled_x,
           labelled_y,
           sampled_global_latents=None,
           attentive_encodings=None,
           lengthscale_1=1.,
           lengthscale_2=1.):

    def _get_mean_var(inputs):
      logits = self._net(inputs)
      mean, untransformed_var = tf.split(logits, 2, axis=-1)
      if self._output_activation is not None:
        mean = self._output_activation(mean)
      var = tf.nn.softplus(untransformed_var)
      return mean, var

    tiled_unlabelled_dataset_encoding = tf.tile(
        sampled_global_latents,
        [1, tf.shape(unlabelled_x)[1], 1])
    tiled_labelled_dataset_encoding = tf.tile(
        sampled_global_latents,
        [1, tf.shape(labelled_x)[1], 1])

    if self._uncertainty_type == 'attentive_gp':
      if self._net is not None:
        unlabelled_inputs = tf.concat(
            [unlabelled_x, tiled_unlabelled_dataset_encoding], axis=-1)
        global_unlabelled_mean, _ = _get_mean_var(unlabelled_inputs)
        labelled_inputs = tf.concat(
            [labelled_x, tiled_labelled_dataset_encoding], axis=-1)
        global_labelled_mean, _ = _get_mean_var(labelled_inputs)
      else:
        global_unlabelled_mean = 0.
        global_labelled_mean = 0.

      mean = global_unlabelled_mean + self._attention_mean(
          unlabelled_x,
          labelled_x,
          labelled_y-global_labelled_mean,
          normalize=True,
          scale=lengthscale_1)
      k_xx = 1.
      k_xd = self._attention_scale_1(
          unlabelled_x,
          labelled_x,
          scale=lengthscale_2,
          normalize=True,
          weights_only=True)
      w_xd = self._attention_scale_2(
          unlabelled_x,
          labelled_x,
          scale=lengthscale_1,
          normalize=True,
          weights_only=True)
      var = k_xx - tf.reduce_sum(
          w_xd * k_xd, axis=-1, keepdims=True)
    else:
      inputs = tf.concat([unlabelled_x,
                          tiled_unlabelled_dataset_encoding,
                          attentive_encodings], axis=-1)
      mean, var = _get_mean_var(inputs)

    std = tf.sqrt(var + eps)
    return ed.Normal(loc=mean, scale=std)
