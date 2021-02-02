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

"""Bayesian linear model."""

from edward2.tensorflow import generated_random_variables

import tensorflow as tf


class BayesianLinearModel(tf.python.keras.Model):
  r"""Bayesian linear model with standard normal prior over its coefficients.

  A forward pass computes the mean of the exact predictive distribution

  ```none
  p(outputs | inputs) = \int Normal(outputs | coeffs * inputs, noise_variance)
                             Normal(coeffs | 0, 1) dweights dbias.
  ```

  It takes a Tensor of shape [batch_size, input_dim] as input and returns a
  Normal random variable of shape [batch_size] representing its outputs.
  After `fit()`, the forward pass computes the exact posterior predictive
  distribution.
  """

  def __init__(self, noise_variance, **kwargs):
    super(BayesianLinearModel, self).__init__(**kwargs)
    self.noise_variance = noise_variance
    self.coeffs_precision_tril_op = None
    self.coeffs_mean = None

  def call(self, inputs):
    if self.coeffs_mean is None and self.coeffs_precision_tril_op is None:
      # p(mean(ynew) | xnew) = Normal(ynew | mean = 0, variance = xnew xnew^T)
      predictive_mean = 0.
      predictive_variance = tf.reduce_sum(tf.square(inputs), axis=-1)
    else:
      # p(mean(ynew) | xnew, x, y) = Normal(ynew |
      #   mean = xnew (1/noise_variance) (1/noise_variance x^T x + I)^{-1}x^T y,
      #   variance = xnew (1/noise_variance x^T x + I)^{-1} xnew^T)
      predictive_mean = tf.einsum('nm,m->n', inputs, self.coeffs_mean)
      predictive_covariance = tf.matmul(
          inputs,
          self.coeffs_precision_tril_op.solve(
              self.coeffs_precision_tril_op.solve(inputs, adjoint_arg=True),
              adjoint=True))
      predictive_variance = tf.linalg.tensor_diag_part(predictive_covariance)
    return generated_random_variables.Normal(loc=predictive_mean,
                                             scale=tf.sqrt(predictive_variance))

  def fit(self, x=None, y=None):
    # p(coeffs | x, y) = Normal(coeffs |
    #   mean = (1/noise_variance) (1/noise_variance x^T x + I)^{-1} x^T y,
    #   covariance = (1/noise_variance x^T x + I)^{-1})
    # TODO(trandustin): We newly fit the data at each call. Extend to do
    # Bayesian updating.
    kernel_matrix = tf.matmul(x, x, transpose_a=True) / self.noise_variance
    coeffs_precision = tf.linalg.set_diag(
        kernel_matrix, tf.linalg.diag_part(kernel_matrix) + 1.)
    coeffs_precision_tril = tf.linalg.cholesky(coeffs_precision)
    self.coeffs_precision_tril_op = tf.linalg.LinearOperatorLowerTriangular(
        coeffs_precision_tril)
    self.coeffs_mean = self.coeffs_precision_tril_op.solvevec(
        self.coeffs_precision_tril_op.solvevec(tf.einsum('nm,n->m', x, y)),
        adjoint=True) / self.noise_variance
    # TODO(trandustin): To be fully Keras-compatible, return History object.
    return
