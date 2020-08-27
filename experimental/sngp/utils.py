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
import tensorflow as tf


def mean_field_logits(logits, covmat, mean_field_factor=1.):
  """Adjust the predictive logits so its softmax approximates posterior mean."""
  logits_scale = tf.sqrt(1. + tf.linalg.diag_part(covmat) * mean_field_factor)
  if mean_field_factor > 0:
    logits = logits / tf.expand_dims(logits_scale, axis=-1)

  return logits
