# coding=utf-8
# Copyright 2020 The Edward2 Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.experimental.auxiliary_sampling.sampling import mean_field_fn

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def conv_net(n_examples, input_shape, num_classes):
  """Build a simple, feed forward Bayesian neural net."""
  p_fn, q_fn = mean_field_fn(empirical_bayes=True)

  def normalized_kl_fn(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.to_float(n_examples)

  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = tfp.layers.Convolution2DFlipout(
      6,
      kernel_size=5,
      padding="SAME",
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          inputs)
  pool1 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding="SAME")(
          conv1)
  conv2 = tfp.layers.Convolution2DFlipout(
      16,
      kernel_size=5,
      padding="SAME",
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          pool1)
  pool2 = tf.keras.layers.MaxPooling2D(
      pool_size=[2, 2], strides=[2, 2], padding="SAME")(
          conv2)
  conv3 = tfp.layers.Convolution2DFlipout(
      120,
      kernel_size=5,
      padding="SAME",
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = tfp.layers.DenseLocalReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          flatten)
  dense2 = tfp.layers.DenseLocalReparameterization(
      num_classes,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(
          dense1)

  output_dist = tfp.layers.DistributionLambda(
      lambda o: tfd.Categorical(logits=o))(
          dense2)
  return tf.keras.models.Model(inputs=inputs, outputs=output_dist)
