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

"""Initializers.

This module extends `tf.python.keras.initializers` with the notion of "trainable
initializers", where initializers to weights and biases in `tf.python.keras.layers` may
themselves carry parameters. For example, consider a weight initializer which
returns a variational distribution: this is reified as an `ed.RandomVariable`
parameterized by `tf.Variables`.

One subtlety is how `tf.python.keras.constraints` are used on the parameters of
trainable initializers. Typically, Keras constraints are used with projected
gradient descent, where one performs unconstrained optimization and then applies
a projection (the constraint) after each gradient update. To stay in line with
probabilistic literature, trainable initializers apply constraints on the
`tf.Variables` themselves (i.e., a constrained parameterization) and do not
apply projections during optimization.

## References:

[1]: Felix Xinnan Yu et al. Orthogonal Random Features. In _Neural Information
     Processing Systems_, 2016.
     https://papers.nips.cc/paper/6246-orthogonal-random-features.pdf
"""

import math

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables
from edward2.tensorflow import regularizers

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def get_condconv_initializer(initializer, num_experts, expert_shape):
  """Wraps the initializer to correctly initialize CondConv variables.

  CondConv initializes biases and kernels in a num_experts x num_params
  matrix for efficient computation. This wrapper ensures that each expert
  is correctly initialized with the given initializer before being flattened
  into the correctly shaped CondConv variable.

  Arguments:
    initializer: The initializer to apply for each individual expert.
    num_experts: The number of experts to be initialized.
    expert_shape: The original shape of each individual expert.

  Returns:
    The initializer for the num_experts x num_params CondConv variable.
  """
  def condconv_initializer(expected_shape, dtype=None, partition=None):
    """CondConv initializer function."""
    num_params = np.prod(expert_shape)
    if (len(expected_shape) != 2 or expected_shape[0] != num_experts or
        expected_shape[1] != num_params):
      raise (ValueError(
          'CondConv variables must have shape [num_experts, num_params]'))
    flattened_kernels = []
    for _ in range(num_experts):
      if partition is None:  # partition is not defined for a given initializer
        kernel = initializer(expert_shape, dtype)
      else:
        kernel = initializer(expert_shape, dtype, partition)
      flattened_kernels.append(tf.reshape(kernel, [-1]))
    return tf.stack(flattened_kernels)

  return condconv_initializer


# From `tensorflow/python/ops/init_ops.py`
def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1.
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return fan_in, fan_out


class ScaledNormalStdDev(tf.python.keras.initializers.VarianceScaling):
  """Initializer capable of adapting its scale to the shape of weights tensors.

  This initializes the standard deviation parameter of a Trainable Normal
  distribution with a scale based on the shape of the weights tensor.
  Additionally, A small amount of noise will be added to break weigh symmetry.

  With `distribution="truncated_normal" or "untruncated_normal"`, the standard
  deviation (after truncation, if used) is `stddev = sqrt(scale / n)`, where n
  is:
    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"
  """

  def __init__(self,
               scale=1.0,
               mode='fan_in',
               distribution='untruncated_normal',
               seed=None):
    """Constructs the initializer.

    Args:
      scale: Scaling factor (positive float).
      mode: One of "fan_in", "fan_out", "fan_avg".
      distribution: Random distribution to use. One of "truncated_normal", or
        "untruncated_normal".
      seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed`
        for behavior.

    Raises:
      ValueError: In case of an invalid value for the "scale", mode" or
        "distribution" arguments.
    """
    distribution = distribution.lower()
    if distribution not in {'truncated_normal', 'untruncated_normal'}:
      raise ValueError('Invalid `distribution` argument:', distribution)
    super(ScaledNormalStdDev, self).__init__(scale=scale, mode=mode,
                                             distribution=distribution,
                                             seed=seed)

  def __call__(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    fan_in, fan_out = _compute_fans(scale_shape)
    if self.mode == 'fan_in':
      scale /= max(1., fan_in)
    elif self.mode == 'fan_out':
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == 'truncated_normal':
      # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
    else:  # self.distribution == 'untruncated_normal':
      stddev = math.sqrt(scale)
    return tf.random.truncated_normal(shape, mean=stddev, stddev=stddev*0.1,
                                      dtype=dtype, seed=self.seed)


class TrainableDeterministic(tf.python.keras.layers.Layer):
  """Deterministic point-wise initializer with trainable location."""

  def __init__(self,
               loc_initializer='glorot_uniform',
               loc_regularizer=None,
               loc_constraint=None,
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableDeterministic, self).__init__(**kwargs)
    self.loc_initializer = get(loc_initializer)
    self.loc_regularizer = regularizers.get(loc_regularizer)
    self.loc_constraint = constraints.get(loc_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.loc = self.add_weight(
        'loc',
        shape=shape,
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    loc = self.loc
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    return generated_random_variables.Independent(
        generated_random_variables.Deterministic(loc=loc).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'loc_initializer':
            serialize(self.loc_initializer),
        'loc_regularizer':
            regularizers.serialize(self.loc_regularizer),
        'loc_constraint':
            constraints.serialize(self.loc_constraint),
        'seed': self.seed,
    }


class TrainableHalfCauchy(tf.python.keras.layers.Layer):
  """Half-Cauchy distribution initializer with trainable parameters."""

  def __init__(self,
               loc_initializer=tf.python.keras.initializers.TruncatedNormal(
                   stddev=1e-5),
               scale_initializer=tf.python.keras.initializers.TruncatedNormal(
                   mean=-3., stddev=0.1),
               loc_regularizer=None,
               scale_regularizer=None,
               loc_constraint=None,
               scale_constraint='softplus',
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableHalfCauchy, self).__init__(**kwargs)
    self.loc_initializer = get(loc_initializer)
    self.scale_initializer = get(scale_initializer)
    self.loc_regularizer = regularizers.get(loc_regularizer)
    self.scale_regularizer = regularizers.get(scale_regularizer)
    self.loc_constraint = constraints.get(loc_constraint)
    self.scale_constraint = constraints.get(scale_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.loc = self.add_weight(
        'loc',
        shape=shape,
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.scale = self.add_weight(
        'scale',
        shape=shape,
        initializer=self.scale_initializer,
        regularizer=self.scale_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    loc = self.loc
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    scale = self.scale
    if self.scale_constraint:
      scale = self.scale_constraint(scale)
    return generated_random_variables.Independent(
        generated_random_variables.HalfCauchy(loc=loc,
                                              scale=scale).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'loc_initializer':
            serialize(self.loc_initializer),
        'scale_initializer':
            serialize(self.scale_initializer),
        'loc_regularizer':
            regularizers.serialize(self.loc_regularizer),
        'scale_regularizer':
            regularizers.serialize(self.scale_regularizer),
        'loc_constraint':
            constraints.serialize(self.loc_constraint),
        'scale_constraint':
            constraints.serialize(self.scale_constraint),
        'seed': self.seed,
    }


class TrainableCauchy(tf.python.keras.layers.Layer):
  """Cauchy distribution initializer with trainable parameters."""

  def __init__(
      self,
      loc_initializer=tf.python.keras.initializers.TruncatedNormal(stddev=1e-5),
      scale_initializer=tf.python.keras.initializers.TruncatedNormal(
          mean=-3., stddev=0.1),
      loc_regularizer=None,
      scale_regularizer=None,
      loc_constraint=None,
      scale_constraint='softplus',
      seed=None,
      **kwargs):
    """Constructs the initializer."""
    super(TrainableCauchy, self).__init__(**kwargs)
    self.loc_initializer = get(loc_initializer)
    self.scale_initializer = get(scale_initializer)
    self.loc_regularizer = regularizers.get(loc_regularizer)
    self.scale_regularizer = regularizers.get(scale_regularizer)
    self.loc_constraint = constraints.get(loc_constraint)
    self.scale_constraint = constraints.get(scale_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.loc = self.add_weight(
        'loc',
        shape=shape,
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.scale = self.add_weight(
        'scale',
        shape=shape,
        initializer=self.scale_initializer,
        regularizer=self.scale_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    loc = self.loc
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    scale = self.scale
    if self.scale_constraint:
      scale = self.scale_constraint(scale)
    return generated_random_variables.Independent(
        generated_random_variables.Cauchy(loc=loc, scale=scale).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'loc_initializer': serialize(self.loc_initializer),
        'scale_initializer': serialize(self.scale_initializer),
        'loc_regularizer': regularizers.serialize(self.loc_regularizer),
        'scale_regularizer': regularizers.serialize(self.scale_regularizer),
        'loc_constraint': constraints.serialize(self.loc_constraint),
        'scale_constraint': constraints.serialize(self.scale_constraint),
        'seed': self.seed,
    }


class TrainableLogNormal(tf.python.keras.layers.Layer):
  """Random log normal op as an initializer with trainable loc and scale."""

  def __init__(self,
               loc_initializer=tf.python.keras.initializers.TruncatedNormal(
                   stddev=1e-5),
               scale_initializer=tf.python.keras.initializers.TruncatedNormal(
                   mean=-3., stddev=0.1),
               loc_regularizer=None,
               scale_regularizer=None,
               loc_constraint=None,
               scale_constraint='softplus',
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableLogNormal, self).__init__(**kwargs)
    self.loc_initializer = get(loc_initializer)
    self.scale_initializer = get(scale_initializer)
    self.loc_regularizer = regularizers.get(loc_regularizer)
    self.scale_regularizer = regularizers.get(scale_regularizer)
    self.loc_constraint = constraints.get(loc_constraint)
    self.scale_constraint = constraints.get(scale_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.loc = self.add_weight(
        'loc',
        shape=shape,
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.scale = self.add_weight(
        'scale',
        shape=shape,
        initializer=self.scale_initializer,
        regularizer=self.scale_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    loc = self.loc
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    scale = self.scale
    if self.scale_constraint:
      scale = self.scale_constraint(scale)
    return generated_random_variables.Independent(
        generated_random_variables.LogNormal(loc=loc, scale=scale).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'loc_initializer':
            serialize(self.loc_initializer),
        'scale_initializer':
            serialize(self.scale_initializer),
        'loc_regularizer':
            regularizers.serialize(self.loc_regularizer),
        'scale_regularizer':
            regularizers.serialize(self.scale_regularizer),
        'loc_constraint':
            constraints.serialize(self.loc_constraint),
        'scale_constraint':
            constraints.serialize(self.scale_constraint),
        'seed': self.seed,
    }


class TrainableNormal(tf.python.keras.layers.Layer):
  """Random normal op as an initializer with trainable mean and stddev."""

  def __init__(self,
               mean_initializer=tf.python.keras.initializers.TruncatedNormal(
                   stddev=1e-5),
               stddev_initializer=tf.python.keras.initializers.TruncatedNormal(
                   mean=-3., stddev=0.1),
               mean_regularizer=None,
               stddev_regularizer=None,
               mean_constraint=None,
               stddev_constraint='softplus',
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableNormal, self).__init__(**kwargs)
    self.mean_initializer = get(mean_initializer)
    self.stddev_initializer = get(stddev_initializer)
    self.mean_regularizer = regularizers.get(mean_regularizer)
    self.stddev_regularizer = regularizers.get(stddev_regularizer)
    self.mean_constraint = constraints.get(mean_constraint)
    self.stddev_constraint = constraints.get(stddev_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.mean = self.add_weight(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.stddev = self.add_weight(
        'stddev',
        shape=shape,
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    mean = self.mean
    if self.mean_constraint:
      mean = self.mean_constraint(mean)
    stddev = self.stddev
    if self.stddev_constraint:
      stddev = self.stddev_constraint(stddev)
    return generated_random_variables.Independent(
        generated_random_variables.Normal(loc=mean, scale=stddev).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'mean_initializer':
            serialize(self.mean_initializer),
        'stddev_initializer':
            serialize(self.stddev_initializer),
        'mean_regularizer':
            regularizers.serialize(self.mean_regularizer),
        'stddev_regularizer':
            regularizers.serialize(self.stddev_regularizer),
        'mean_constraint':
            constraints.serialize(self.mean_constraint),
        'stddev_constraint':
            constraints.serialize(self.stddev_constraint),
        'seed': self.seed,
    }


class TrainableHeNormal(TrainableNormal):
  """Trainable normal initialized per He et al. 2015, given a ReLU nonlinearity.

  The distribution is initialized to a Normal scaled by `sqrt(2 / fan_in)`,
  where `fan_in` is the number of input units. A ReLU nonlinearity is assumed
  for this initialization scheme.

  References:
    He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification. In Proceedings of the
    IEEE international conference on computer vision 2015 (pp. 1026-1034).
    https://arxiv.org/abs/1502.01852
  """

  def __init__(self, seed=None, **kwargs):
    super(TrainableHeNormal, self).__init__(
        mean_initializer=tf.python.keras.initializers.he_normal(seed),
        seed=seed,
        **kwargs)

  def get_config(self):
    return {
        'seed': self.seed,
    }


class TrainableGlorotNormal(TrainableNormal):
  """Trainable normal initialized per Glorot and Bengio, 2010.

  The distribution is initialized to a Normal scaled by `sqrt(2 / fan_in +
  fan_out)`, where `fan_in` is the number of input units and `fan_out` is the
  number of output units.

  References:
    Glorot X, Bengio Y. Understanding the difficulty of training deep
    feedforward neural networks. In Proceedings of the thirteenth international
    conference on artificial intelligence and statistics 2010 Mar 31 (pp.
    249-256). http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  """

  def __init__(self, seed=None, **kwargs):
    super(TrainableGlorotNormal, self).__init__(
        mean_initializer=tf.python.keras.initializers.GlorotNormal(seed),
        seed=seed,
        **kwargs)

  def get_config(self):
    return {
        'seed': self.seed,
    }


class TrainableNormalSharedStddev(TrainableNormal):
  """Random normal op as an initializer with trainable mean and stddev.

  The stddev parameter is a scalar shared across the weight. This enables, e.g.,
  learnable dropout rates per-layer during Gaussian variational dropout rather
  than learnable dropout rates per-weight.
  """

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.mean = self.add_weight(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.stddev = self.add_weight(
        'stddev',
        shape=(),
        initializer=self.stddev_initializer,
        regularizer=self.stddev_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True


class TrainableNormalFixedStddev(tf.python.keras.layers.Layer):
  """Random normal op as an initializer with trainable mean and fixed stddev."""

  def __init__(self,
               stddev=1.,
               mean_initializer=tf.python.keras.initializers.TruncatedNormal(
                   stddev=1e-5),
               mean_regularizer=None,
               mean_constraint=None,
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableNormalFixedStddev, self).__init__(**kwargs)
    self.stddev = stddev
    self.mean_initializer = get(mean_initializer)
    self.mean_regularizer = regularizers.get(mean_regularizer)
    self.mean_constraint = constraints.get(mean_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype
    self.mean = self.add_weight(
        'mean',
        shape=shape,
        initializer=self.mean_initializer,
        regularizer=self.mean_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    mean = self.mean
    if self.mean_constraint:
      mean = self.mean_constraint(mean)
    return generated_random_variables.Independent(
        generated_random_variables.Normal(loc=mean,
                                          scale=self.stddev).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'stddev': self.stddev,
        'mean_initializer': serialize(self.mean_initializer),
        'mean_regularizer': regularizers.serialize(self.mean_regularizer),
        'mean_constraint': constraints.serialize(self.mean_constraint),
        'seed': self.seed,
    }


class RandomSign(tf.python.keras.initializers.Initializer):
  """Initializer that generates tensors initialized to +/- 1.

  Attributes:
    probs: probability of +1.
    dtype: tensorflow dtype.
    seed: A Python integer. Used to create random seeds. See
      `tf.set_random_seed`
  """

  def __init__(self, probs=1.0, seed=None, dtype=tf.float32):
    self.probs = probs
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    bernoulli = tfp.distributions.Bernoulli(probs=self.probs,
                                            dtype=dtype)
    return 2. * bernoulli.sample(shape, self.seed) - 1.

  def get_config(self):
    return {
        'dtype': self.dtype.name,
        'seed': self.seed,
        'probs': self.probs
    }


class TrainableMixtureOfDeltas(tf.python.keras.layers.Layer):
  """Mixture of deltas as an initializer with trainable locations."""

  def __init__(self,
               num_components=5,
               loc_initializer=tf.python.keras.initializers.he_normal(),
               loc_regularizer=None,
               loc_constraint=None,
               seed=None,
               **kwargs):
    """Constructs the initializer."""
    super(TrainableMixtureOfDeltas, self).__init__(**kwargs)
    self.num_components = num_components
    self.loc_initializer = get(loc_initializer)
    self.loc_regularizer = regularizers.get(loc_regularizer)
    self.loc_constraint = constraints.get(loc_constraint)
    self.seed = seed

  def build(self, shape, dtype=None):
    if dtype is None:
      dtype = self.dtype

    self.loc = self.add_weight(
        'loc',
        shape=list(shape) + [self.num_components],
        initializer=self.loc_initializer,
        regularizer=self.loc_regularizer,
        constraint=None,
        dtype=dtype,
        trainable=True)
    self.built = True

  def __call__(self, shape, dtype=None):
    if not self.built:
      self.build(shape, dtype)
    loc = self.loc
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    return generated_random_variables.Independent(
        generated_random_variables.MixtureSameFamily(
            mixture_distribution=generated_random_variables.Categorical(
                probs=tf.broadcast_to(
                    [[1/self.num_components]*self.num_components],
                    list(shape) + [self.num_components])).distribution,
            components_distribution=generated_random_variables.Deterministic(
                loc=loc).distribution
        ).distribution,
        reinterpreted_batch_ndims=len(shape))

  def get_config(self):
    return {
        'num_components': self.num_components,
        'loc_initializer':
            serialize(self.loc_initializer),
        'loc_regularizer':
            regularizers.serialize(self.loc_regularizer),
        'loc_constraint':
            constraints.serialize(self.loc_constraint),
        'seed': self.seed,
    }


class OrthogonalRandomFeatures(tf.python.keras.initializers.Orthogonal):
  """Generates a orthogonal Gaussian matrix for a random feature Dense layer.

  Generates a 2D matrix of form W = stddev * Q @ S [1], where Q is a random
  orthogonal matrix of shape (num_rows, num_cols), and S is a diagonal matrix
  of i.i.d. random variables following chi(df = num_rows) distribution that
  imitates the column norm of a random Gaussian matrix.
  """

  def __init__(self, stddev=1.0, random_norm=True, seed=None):
    """Initializer.

    Args:
      stddev: (float) The standard deviation of the random matrix.
      random_norm: (bool) Whether to sample the norms of the random matrix
        columns from a chi(df=num_cols) distribution, or fix it to
        sqrt(num_cols). These two options corresponds to the construction in
        Theorem 1 and Theorem 2 of [1].
      seed: (int) Random seed.
    """
    super(OrthogonalRandomFeatures, self).__init__(gain=stddev, seed=seed)
    self.stddev = stddev
    self.random_norm = random_norm

  def _sample_orthogonal_matrix(self, shape, dtype):
    return super(OrthogonalRandomFeatures, self).__call__(shape, dtype=dtype)

  def __call__(self, shape, dtype=tf.float32, **kwargs):
    # Sample orthogonal matrices.
    num_rows, num_cols = shape
    if num_rows < num_cols:
      # When num_row < num_col, sample multiple (num_row, num_row) matrices and
      # then concatenate following [1].
      ortho_mat_list = []
      num_cols_sampled = 0

      while num_cols_sampled < num_cols:
        ortho_mat_square = self._sample_orthogonal_matrix(
            (num_rows, num_rows), dtype=dtype)
        ortho_mat_list.append(ortho_mat_square)
        num_cols_sampled += num_rows

      # Reshape the matrix to the target shape (num_rows, num_cols)
      ortho_mat = tf.concat(ortho_mat_list, axis=-1)
      ortho_mat = ortho_mat[:, :num_cols]
    else:
      ortho_mat = self._sample_orthogonal_matrix(shape, dtype=dtype)

    # Sample random feature norms.
    if self.random_norm:
      # Construct Monte-Carlo estimate of squared column norm of a random
      # Gaussian matrix.
      feature_norms_square = tf.random.normal(shape=ortho_mat.shape)**2
    else:
      # Use mean of the squared column norm (i.e., E(z**2)=1) instead.
      feature_norms_square = tf.ones(shape=ortho_mat.shape)

    feature_norms = tf.reduce_sum(feature_norms_square, axis=0)
    feature_norms = tf.sqrt(feature_norms)

    # Returns the random feature matrix with orthogonal column and Gaussian-like
    # column norms.
    return ortho_mat * feature_norms

  def get_config(self):
    config = {
        'stddev': self.stddev,
        'random_norm': self.random_norm,
    }
    new_config = super(OrthogonalRandomFeatures, self).get_config()
    config.update(new_config)
    return config

# Compatibility aliases, following tf.python.keras

# pylint: disable=invalid-name
scaled_normal_std_dev = ScaledNormalStdDev
trainable_deterministic = TrainableDeterministic
trainable_half_cauchy = TrainableHalfCauchy
trainable_cauchy = TrainableCauchy
trainable_normal = TrainableNormal
trainable_he_normal = TrainableHeNormal
trainable_glorot_normal = TrainableGlorotNormal
trainable_log_normal = TrainableLogNormal
trainable_normal_shared_stddev = TrainableNormalSharedStddev
trainable_normal_fixed_stddev = TrainableNormalFixedStddev
trainable_mixture_of_deltas = TrainableMixtureOfDeltas
random_sign = RandomSign
orthogonal_random_features = OrthogonalRandomFeatures
# pylint: enable=invalid-name

# Utility functions, following tf.python.keras


def serialize(initializer):
  return tf.python.keras.utils.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.python.keras.utils.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='initializers')


def get(identifier, value=None):
  """Getter for loading from strings; falls back to Keras as needed."""
  if value is None:
    value = identifier
  if identifier is None:
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      pass
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      pass
  elif callable(identifier):
    return identifier
  return tf.python.keras.initializers.get(value)
