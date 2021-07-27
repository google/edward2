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

"""Neural network module in style of Flax."""

from edward2.jax.nn import utils
from edward2.jax.nn.dense import DenseBatchEnsemble
from edward2.jax.nn.heteroscedastic_lib import MCSigmoidDenseFA
from edward2.jax.nn.heteroscedastic_lib import MCSoftmaxDenseFA
from edward2.jax.nn.random_feature import LaplaceRandomFeatureCovariance
from edward2.jax.nn.random_feature import RandomFeatureGaussianProcess
from edward2.jax.nn.random_feature import RandomFourierFeatures

__all__ = [
    "DenseBatchEnsemble",
    "MCSoftmaxDenseFA",
    "MCSigmoidDenseFA",
    "LaplaceRandomFeatureCovariance",
    "RandomFeatureGaussianProcess",
    "RandomFourierFeatures",
    "utils",
]
