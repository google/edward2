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

"""Utilities for UCI datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
from absl import flags
import numpy as np
import pandas as pd
import scipy
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

# TODO(trandustin)
flags.DEFINE_float('mean_field_init_untransformed_scale', -7,
                   'Initial scale (before softplus) for mean field.')
FLAGS = flags.FLAGS


class DataSpec(collections.namedtuple(
    'UCIDataSpec', 'path,desc,label,excluded')):

  __slots__ = []


# TODO(trandustin): Avoid hard-coding directory string so it's user-specified.
UCI_BASE_DIR = '/tmp/uci_datasets'
DATA_SPECS = {
    'boston_housing': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'boston_housing.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='MEDV',
        excluded=[]),
    'concrete_strength': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'concrete_strength.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='concrete_compressive_strength',
        excluded=[]),
    'energy_efficiency': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'energy_efficiency.csv'),
        desc=('This study looked into assessing the heating load and cooling '
              'load requirements of buildings (that is, energy efficiency) as '
              'a function of building parameters. **Heating load only**.'),
        label='Y1',
        excluded=['Y2']),
    'naval_propulsion': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'naval_propulsion.csv'),
        desc=('Data have been generated from a sophisticated simulator of a '
              'Gas Turbines (GT), mounted on a Frigate characterized by a '
              'Combined Diesel eLectric And Gas (CODLAG) propulsion plant '
              'type. **GT Turbine decay state coefficient only**'),
        label='GT Turbine decay state coefficient',
        excluded=['GT Compressor decay state coefficient']),
    'kin8nm': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'kin8nm.csv'),
        desc=('This is data set is concerned with the forward kinematics of '
              'an 8 link robot arm. Among the existing variants of this data '
              'set we have used the variant 8nm, which is known to be highly '
              'non-linear and medium noisy.'),
        label='y',
        excluded=[]),
    'power_plant': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'power_plant.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='PE',
        excluded=[]),
    'protein_structure': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'protein_structure.csv'),
        desc=('This is a data set of Physicochemical Properties of Protein '
              'Tertiary Structure. The data set is taken from CASP 5-9. There '
              'are 45730 decoys and size varying from 0 to 21 armstrong.'),
        label='RMSD',
        excluded=[]),
    'wine': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'wine.csv'),
        desc=('The dataset is related to red variant of the Portuguese '
              '"Vinho Verde" wine. **NB contains red wine examples only**'),
        label='quality',
        excluded=[]),
    'yacht_hydrodynamics': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'yacht_hydrodynamics.csv'),
        desc=('Delft data set, used to predict the hydodynamic performance of '
              'sailing yachts from dimensions and velocity.'),
        label='Residuary resistance per unit weight of displacement',
        excluded=[])
}


def get_uci_data(name):
  """Returns an array of features and a vector of labels for dataset `name`."""
  spec = DATA_SPECS.get(name)
  if spec is None:
    raise ValueError('Unknown dataset: {}. Available datasets:\n{}'.format(
        name, '\n'.join(DATA_SPECS.keys())))
  with tf.io.gfile.GFile(spec.path) as f:
    df = pd.read_csv(f)
  labels = df.pop(spec.label).as_matrix().astype(np.float32)
  for ex in spec.excluded:
    _ = df.pop(ex)
  features = df.as_matrix().astype(np.float32)
  return features, labels


def load(name):
  """Loads dataset as numpy array."""
  x, y = get_uci_data(name)
  if len(y.shape) == 1:
    y = y[:, None]
  train_test_split = 0.8
  random_permutation = np.random.permutation(x.shape[0])
  n_train = int(x.shape[0] * train_test_split)
  train_ind = random_permutation[:n_train]
  test_ind = random_permutation[n_train:]
  x_train, y_train = x[train_ind, :], y[train_ind, :]
  x_test, y_test = x[test_ind, :], y[test_ind, :]

  x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
  y_mean = np.mean(y_train, axis=0)
  epsilon = tf.keras.backend.epsilon()
  x_train = (x_train - x_mean) / (x_std + epsilon)
  x_test = (x_test - x_mean) / (x_std + epsilon)
  y_train, y_test = y_train - y_mean, y_test - y_mean
  return x_train, y_train, x_test, y_test


def ensemble_metrics(x,
                     y,
                     model,
                     log_likelihood_fn,
                     n_samples=1,
                     weight_files=None):
  """Evaluate metrics of an ensemble.

  Args:
    x: numpy array of inputs
    y: numpy array of labels
    model: tf.keras.Model.
    log_likelihood_fn: keras function of log likelihood.
    n_samples: number of Monte Carlo samples to draw per ensemble member (each
      weight file).
    weight_files: to draw samples from multiple weight sets, specify a list of
      weight files to load. These files must have been generated through
      keras's model.save_weights(...).

  Returns:
    metrics_dict: dictionary containing the metrics
  """
  if weight_files is None:
    ensemble_logprobs = [log_likelihood_fn([x, y])[0] for _ in range(n_samples)]
    metric_values = [model.evaluate(x, y, verbose=0)
                     for _ in range(n_samples)]
    ensemble_error = [log_likelihood_fn([x, y])[1] for _ in range(n_samples)]
  else:
    ensemble_logprobs = []
    metric_values = []
    ensemble_error = []
    for filename in weight_files:
      model.load_weights(filename)
      ensemble_logprobs.extend([
          log_likelihood_fn([x, y])[0] for _ in range(n_samples)])
      ensemble_error.extend([
          log_likelihood_fn([x, y])[1] for _ in range(n_samples)])
      metric_values.extend([
          model.evaluate(x, y, verbose=0) for _ in range(n_samples)])

  metric_values = np.mean(np.array(metric_values), axis=0)
  results = {}
  for m, name in zip(metric_values, model.metrics_names):
    results[name] = m

  ensemble_logprobs = np.array(ensemble_logprobs)
  probabilistic_log_likelihood = np.mean(
      scipy.special.logsumexp(
          np.sum(ensemble_logprobs, axis=2)
          if len(ensemble_logprobs.shape) > 2 else ensemble_logprobs,
          b=1. / ensemble_logprobs.shape[0],
          axis=0),
      axis=0)
  results['probabilistic_log_likelihood'] = probabilistic_log_likelihood
  ensemble_error = np.stack([np.array(l) for l in ensemble_error])
  results['probabilistic_mse'] = np.mean(
      np.square(np.mean(ensemble_error, axis=0)))
  return results


def mean_field_fn(empirical_bayes=False,
                  initializer=tf1.initializers.he_normal()):
  """Constructors for Gaussian prior and posterior distributions.

  Args:
    empirical_bayes (bool): Whether to train the variance of the prior or not.
    initializer (tf1.initializer): Initializer for the posterior means.
  Returns:
    prior, posterior (tfp.distribution): prior and posterior
    to be fed into a Bayesian Layer.
  """

  def prior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the prior distribution (tfp.distributions.Independent)."""
    softplus_inverse_scale = np.log(np.exp(1.) - 1.)

    istrainable = add_variable_fn(
        name=name + '_istrainable',
        shape=(),
        initializer=tf1.constant_initializer(1.),
        dtype=dtype,
        trainable=False)

    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=(),
        initializer=tf1.constant_initializer(softplus_inverse_scale),
        dtype=dtype,
        trainable=empirical_bayes and trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale * istrainable + (1. - istrainable) *
                       tf1.stop_gradient(untransformed_scale)))
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=tf1.constant_initializer(0.),
        dtype=dtype,
        trainable=False)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.istrainable = istrainable
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  def posterior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the posterior distribution (tfp.distributions.Independent)."""
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=shape,
        initializer=tf1.initializers.random_normal(
            mean=FLAGS.mean_field_init_untransformed_scale, stddev=0.1),
        dtype=dtype,
        trainable=trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale))
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        trainable=trainable)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  return prior, posterior


def sample_auxiliary_op(prior, posterior, aux_variance_ratio):
  r"""Sample the auxiliary variable and calculate the conditionals.

  Given a gaussian prior $$\mathcal{N}(\mu_z, \sigma^2_z)$$
  Define auxiliary variables $$z=a_1+a_2$$ with $$a_1=\mathcal{N}(0,
  \sigma_{a_1}^2)$$ and $$a_2=\mathcal{N}(\mu_z, \sigma_{a_2}^2)$$ with
  $$\frac{\sigma_{a_1}^2}{\sigma^2_z}=$$aux_variance_ratio and
  $$\sigma_{a_1}^2+\sigma_{a_2}^2=\sigma_z^2$$.
  From this, we can calculate the posterior of a1 and the conditional of z.

  Conditional:
  $$p(a_1|z) =  \mathcal{N}(z \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_z^2})$$

  Posterior of $$a_1$$:
  $$q(a_1) =\mathcal{N}(\mu_{q(z)} \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{q(z)}^2\sigma_{a_1}^4}{\sigma_{z}^4} +
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_{z}^2})$$

  Conditional posterior:
  $$q(z|a_1)=\frac{q(a_1|z)q(z)}{q(a_1)}$$

  $$q(z|a_1)=\mathcal{N}(\frac{a_1\sigma^2_{q(z)}\sigma^2_{z} +
  \mu_{q(z)}\sigma^2_{a_2}\sigma^2_{z}}{\sigma^2_{q(z)}\sigma^2_{a_1} +
  \sigma^2_z\sigma^2_{a_2}},
  \frac{\sigma^2_{q(z)}\sigma^2_z\sigma^2_{a_2}}{\sigma^2_{a_1}\sigma^2_{q(z)} +
  \sigma^2_{z}\sigma^2_{a_2}})$$.

  Args:
    prior: The prior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    posterior: The posterior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    aux_variance_ratio: Ratio of the variance of the auxiliary variable and the
      prior. The mean of the auxiliary variable is at 0.

  Returns:
    sampling_op: Tensorflow operation that executes the sampling.
    log_density_ratio: Tensor containing the density ratio of the auxiliary
    variable.
  """
  if aux_variance_ratio > 1. or aux_variance_ratio < 0.:
    raise ValueError(
        'The ratio of the variance of the auxiliary variable must be between 0 '
        'and 1.'
    )

  p_a1_loc = tf.zeros_like(prior.loc)
  p_a1_scale = tf.math.sqrt(prior.scale**2 * aux_variance_ratio)
  p_a1 = tfp.distributions.Normal(loc=p_a1_loc, scale=p_a1_scale)
  p_a2_loc = prior.loc
  p_a2_scale = tf.math.sqrt(prior.scale**2 - p_a1_scale**2)
  # q(a1)
  a1_loc = (posterior.loc - prior.loc) * p_a1_scale**2 / prior.scale**2
  a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a1_scale**2 / prior.scale**2 + p_a2_scale**2) *
      p_a1_scale**2 / prior.scale**2)
  q_a1 = tfp.distributions.Normal(loc=a1_loc, scale=a1_scale)
  a1 = q_a1.sample()

  # q(z|a1)
  z_a1_loc = prior.loc + (
      (posterior.loc - prior.loc) * p_a2_scale**2 * prior.scale**2 +
      a1 * posterior.scale**2 * prior.scale**2) / (
          prior.scale**2 * p_a2_scale**2 + posterior.scale**2 * p_a1_scale**2)
  z_a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a2_scale**2 * prior.scale**2) /
      (prior.scale**2 * p_a2_scale**2 + p_a1_scale**2 * posterior.scale**2))

  with tf1.control_dependencies([
      q_a1.loc, q_a1.scale, p_a1.loc, p_a1.scale, a1, p_a2_loc, p_a2_scale,
      z_a1_loc, z_a1_scale
  ]):
    log_density_ratio = q_a1.log_prob(a1) - p_a1.log_prob(a1)
    prior_update = [
        prior.loc.assign(a1 + p_a2_loc),
        prior.untransformed_scale.assign(tfp.math.softplus_inverse(p_a2_scale))
    ]
    posterior_update = [
        posterior.loc.assign(z_a1_loc),
        posterior.untransformed_scale.assign(
            tfp.math.softplus_inverse(z_a1_scale))
    ]
  return [prior_update, posterior_update], tf.reduce_sum(log_density_ratio)
