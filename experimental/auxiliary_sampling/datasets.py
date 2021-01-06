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

"""Dataset loading utility."""

import numpy as np
import tensorflow_datasets as tfds


def load(session):
  """Load cifar10 as numpy array."""
  train_ds = tfds.load(
      'cifar10', split=tfds.Split.TRAIN, batch_size=-1, as_supervised=True)
  x_train, y_train = session.run(train_ds)
  test_ds = tfds.load(
      'cifar10', split=tfds.Split.TEST, batch_size=-1, as_supervised=True)
  x_test, y_test = session.run(test_ds)
  # Standardize data.
  x_mean = np.mean(x_train, axis=(0, 1, 2))
  x_std = np.std(x_train, axis=(0, 1, 2))
  x_train = (x_train - x_mean) / (x_std + 1e-10)
  x_test = (x_test - x_mean) / (x_std + 1e-10)
  return x_train, y_train, x_test, y_test
