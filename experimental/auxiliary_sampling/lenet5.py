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

"""Build a Convolutional Bayesian neural network."""

import edward2 as ed
from experimental.auxiliary_sampling.sampling import mean_field_fn  # local file import

import tensorflow as tf
import tensorflow_probability as tfp


def lenet5(n_examples, input_shape, num_classes):
  """Builds Bayesian LeNet5."""
  p_fn, q_fn = mean_field_fn(empirical_bayes=True)
  def normalized_kl_fn(q, p, _):
    return q.kl_divergence(p) / tf.cast(n_examples, tf.float32)

  inputs = tf.python.keras.layers.Input(shape=input_shape)
  conv1 = tfp.layers.Convolution2DFlipout(
      6,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(inputs)
  pool1 = tf.python.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv1)
  conv2 = tfp.layers.Convolution2DFlipout(
      16,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(pool1)
  pool2 = tf.python.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
  conv3 = tfp.layers.Convolution2DFlipout(
      120,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(pool2)
  flatten = tf.python.keras.layers.Flatten()(conv3)
  dense1 = tfp.layers.DenseLocalReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(flatten)
  dense2 = tfp.layers.DenseLocalReparameterization(
      num_classes,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(dense1)
  outputs = tf.python.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(dense2)
  return tf.python.keras.models.Model(inputs=inputs, outputs=outputs)
