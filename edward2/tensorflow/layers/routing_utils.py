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

"""Routing utils."""
import tensorflow as tf


def rowwise_unsorted_segment_sum(values, indices, n):
  """UnsortedSegmentSum on each row.

  Args:
  values: a `Tensor` with shape `[batch_size, k]`.
  indices: an integer `Tensor` with shape `[batch_size, k]`.
  n: an integer.

  Returns:
  A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
  """
  batch, k = tf.unstack(tf.shape(indices), num=2)
  indices_flat = tf.reshape(indices, [-1]) + tf.cast(
      tf.math.divide(tf.range(batch * k), k) * n, tf.int32)
  ret_flat = tf.math.unsorted_segment_sum(
      tf.reshape(values, [-1]), indices_flat, batch * n)
  return tf.reshape(ret_flat, [batch, n])


def normal_distribution_cdf(x, stddev):
  """Evaluates the CDF of the normal distribution.

  Normal distribution with mean 0 and standard deviation stddev,
  evaluated at x=x.
  input and output `Tensor`s have matching shapes.
  Args:
    x: a `Tensor`
    stddev: a `Tensor` with the same shape as `x`.

  Returns:
    a `Tensor` with the same shape as `x`.
  """
  return 0.5 * (1.0 + tf.erf(x / (tf.math.sqrt(2) * stddev + 1e-20)))


def prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values,
                  k):
  """Helper function to NoisyTopKGating.

  Computes the probability that value is in top k, given different random noise.
  This gives us a way of backpropagating from a loss that balances the number
  of times each expert is in the top k experts per example.
  In the case of no noise, pass in None for noise_stddev, and the result will
  not be differentiable.
  Args:
    clean_values: a `Tensor` of shape [batch, n].
    noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
      normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a `Tensor` of shape [batch, n], or None
    noisy_top_values: a `Tensor` of shape [batch, m]. "values" Output of
      tf.top_k(noisy_top_values, m).  m >= k+1
    k: an integer.

  Returns:
    a `Tensor` of shape [batch, n].
  """
  batch = tf.shape(clean_values)[0]
  m = tf.shape(noisy_top_values)[1]
  top_values_flat = tf.reshape(noisy_top_values, [-1])
  # we want to compute the threshold that a particular value would have to
  # exceed in order to make the top k.  This computation differs depending
  # on whether the value is already in the top k.
  threshold_positions_if_in = tf.range(batch) * m + k
  threshold_if_in = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_in), 1)
  is_in = tf.greater(noisy_values, threshold_if_in)
  if noise_stddev is None:
    return tf.to_float(is_in)
  threshold_positions_if_out = threshold_positions_if_in - 1
  threshold_if_out = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_out), 1)
  # is each value currently in the top k.
  prob_if_in = normal_distribution_cdf(clean_values - threshold_if_in,
                                       noise_stddev)
  prob_if_out = normal_distribution_cdf(clean_values - threshold_if_out,
                                        noise_stddev)
  prob = tf.where(is_in, prob_if_in, prob_if_out)
  return prob
