# coding=utf-8
# Copyright 2023 The Edward2 Authors.
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

"""Sweep over BatchEnsemble + CAMixup."""


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  sweep = hyper.product([
      hyper.sweep('seed', hyper.discrete(range(3))),
      hyper.sweep('l2', hyper.discrete([1e-4, 2e-4, 3e-4])),
      hyper.sweep('adaptive_mixup', hyper.categorical([False, True])),
      # hyper.sweep('fast_weight_lr_multiplier', hyper.discrete([0.5, 1.0])),
      # hyper.sweep('random_sign_init', hyper.discrete([-0.5, -0.75])),
      # hyper.sweep('per_core_batch_size', hyper.discrete([64, 128])),
  ])
  return sweep
