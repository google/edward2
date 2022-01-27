# coding=utf-8
# Copyright 2022 The Edward2 Authors.
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

"""Study for L1 and L2 regularization paths for CIFAR 10 and CIFAR 100.
"""


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  # pylint: disable=line-too-long
  hyper_dict = {
      'l1': [0.],
      'l2': [1e-5, 5e-5, 1e-4, 5e-3, 1e-3, 0.002186, 0.004781, 0.010456, 0.022865, 0.05],
      'ensemble_size': [1, 2, 3, 4, 5, 6],
  }
  # hyper_dict = {
  #     'l1': [0.],
  #     'l2': [1e-4, 5e-4, 1e-3, 5e-2, 1e-2, 0.02186, 0.04781, 0.10456, 0.22865, 0.5],
  #     'ensemble_size': [1, 2, 3, 4, 5, 6],
  # }
  # hyper_dict = {
  #     'l2': [0.],
  #     'l1': [5e-7, 1e-6, 5e-6, 1e-5, 5e-4, 1e-4, 0.0002186, 0.0004781, 0.0010456, 0.0022865, 0.005],
  #     'ensemble_size': [1, 2, 3, 4, 5, 6],
  #     'dataset': ['cifar100', 'cifar10'],
  # }
  # pylint: enable=line-too-long

  domain = []
  for par_name, par_values in hyper_dict.items():
    is_categorical = not all(
        isinstance(value, (int, float)) for value in par_values)
    if is_categorical:
      hyper_func = hyper.categorical
    else:
      hyper_func = hyper.discrete
    hyper_sweep_spec = hyper.sweep(par_name, hyper_func(par_values))
    domain.append(hyper_sweep_spec)

  sweep = hyper.product(domain)
  return sweep
