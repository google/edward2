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

"""SNGP utilities."""
import warnings
import edward2 as ed


def mean_field_logits(*args, **kwargs):
  """Adjust the predictive logits so its softmax approximates posterior mean."""
  warnings.warn(
      'Please use `edward2.layers.utils.mean_field_logits` instead.',
      category=DeprecationWarning, stacklevel=2)

  return ed.layers.utils.mean_field_logits(*args, **kwargs)
