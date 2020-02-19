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

"""Dataset loading utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np

import pandas as pd
import tensorflow.compat.v1 as tf  # tf
import tensorflow_datasets as tfds


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
  with tf.gfile.Open(spec.path) as f:
    df = pd.read_csv(f)
  labels = df.pop(spec.label).as_matrix().astype(np.float32)
  for ex in spec.excluded:
    _ = df.pop(ex)
  features = df.as_matrix().astype(np.float32)
  return features, labels


def load(name, session):
  """Load dataset (UCI or mnist, fashion_mnist, cifar10) as numpy array."""

  if name in ['mnist', 'fashion_mnist', 'cifar10']:
    train_ds = tfds.load(
        name, split=tfds.Split.TRAIN, batch_size=-1, as_supervised=True)
    x_train, y_train = session.run(train_ds)
    n_train = x_train.shape[0]
    test_ds = tfds.load(
        name, split=tfds.Split.TEST, batch_size=-1, as_supervised=True)
    x_test, y_test = session.run(test_ds)
    # Cifar10 is standardized. The other datasets are scaled to the [0, 1]
    # interval.
    if name == 'cifar10':
      x_mean = np.mean(x_train, axis=(0, 1, 2))
      x_std = np.std(x_train, axis=(0, 1, 2))
      x_train = (x_train - x_mean) / (x_std + 1e-10)
      x_test = (x_test - x_mean) / (x_std + 1e-10)
    else:
      x_min, x_max = np.amin(x_train), np.amax(x_train)
      x_train = (x_train - x_min) / (x_max + 1e-10)
      x_test = (x_test - x_min) / (x_max + 1e-10)
  else:
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

    # Standardize
    x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    y_mean = np.mean(y_train, axis=0)
    x_train = (x_train - x_mean) / (x_std + 1e-10)
    x_test = (x_test - x_mean) / (x_std + 1e-10)
    y_train, y_test = y_train - y_mean, y_test - y_mean
  return x_train, y_train, x_test, y_test
