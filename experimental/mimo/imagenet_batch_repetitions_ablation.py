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

"""Ablation study for choice of batch_repetitions, each setting somewhat tuned.
"""


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  # Sweep over batch repetitions, adjusting each setting so the models always
  # run with the same number of training iterations and learning rate.
  batch_repetitions_list = [1, 2, 3, 4]

  train_epochs_list = [int(150 * 2 / x) for x in batch_repetitions_list]
  lr_decay_epochs_list = []
  for x in batch_repetitions_list:
    lr_decay_epochs_list.append(
        [str(int(30 * 2 / x)), str(int(60 * 2 / x)), str(int(80 * 2 / x))])

  base_learning_rate_list = [0.1 / (2 / x) for x in batch_repetitions_list]

  domain = []
  for [batch_repetitions,
       train_epochs,
       lr_decay_epochs,
       base_learning_rate] in zip(batch_repetitions_list,
                                  train_epochs_list,
                                  lr_decay_epochs_list,
                                  base_learning_rate_list):
    subdomain = [
        hyper.sweep('ensemble_size', hyper.discrete([2, 3])),
        hyper.sweep('l2', hyper.discrete([1e-4, 2e-4, 3e-4])),
        hyper.fixed('batch_repetitions', batch_repetitions, length=1),
        hyper.fixed('train_epochs', train_epochs, length=1),
        hyper.fixed('base_learning_rate', base_learning_rate, length=1),
        hyper.fixed('lr_decay_epochs', lr_decay_epochs, length=1),
    ]
    domain += [hyper.product(subdomain)]

  sweep = hyper.chainit(domain)
  return sweep
