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

"""Ablation study for choice of batch_repetitions, each setting somewhat tuned.
"""


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  batch_repetitions_list = [1, 2, 4, 6]
  base_learning_rate_list = [0.1/4, 0.1/2, 0.1, 0.1*2]
  domain = []
  for [batch_repetitions,
       base_learning_rate] in zip(batch_repetitions_list,
                                  base_learning_rate_list):
    subdomain = [
        hyper.sweep('per_core_batch_size', hyper.discrete([64, 128])),
        hyper.sweep('ensemble_size', hyper.discrete([2, 4])),
        hyper.sweep('l2', hyper.discrete([1e-4, 3e-4, 5e-4])),
        hyper.fixed('batch_repetitions', batch_repetitions, length=1),
        hyper.fixed('base_learning_rate', base_learning_rate, length=1),
    ]
    domain += [hyper.product(subdomain)]

  sweep = hyper.chainit(domain)
  return sweep
