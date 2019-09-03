# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

"""Layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from edward2.tensorflow.layers import utils
from edward2.tensorflow.layers.bayesian_linear_model import BayesianLinearModel
from edward2.tensorflow.layers.convolutional import Conv2DFlipout
from edward2.tensorflow.layers.convolutional import Conv2DHierarchical
from edward2.tensorflow.layers.convolutional import Conv2DReparameterization
from edward2.tensorflow.layers.convolutional import Conv2DVariationalDropout
from edward2.tensorflow.layers.dense import DenseDVI
from edward2.tensorflow.layers.dense import DenseFlipout
from edward2.tensorflow.layers.dense import DenseHierarchical
from edward2.tensorflow.layers.dense import DenseReparameterization
from edward2.tensorflow.layers.dense import DenseVariationalDropout
from edward2.tensorflow.layers.discrete_flows import DiscreteAutoregressiveFlow
from edward2.tensorflow.layers.discrete_flows import DiscreteBipartiteFlow
from edward2.tensorflow.layers.discrete_flows import Reverse
from edward2.tensorflow.layers.discrete_flows import SinkhornAutoregressiveFlow
from edward2.tensorflow.layers.gaussian_process import ExponentiatedQuadratic
from edward2.tensorflow.layers.gaussian_process import GaussianProcess
from edward2.tensorflow.layers.gaussian_process import LinearKernel
from edward2.tensorflow.layers.gaussian_process import SparseGaussianProcess
from edward2.tensorflow.layers.gaussian_process import Zeros
from edward2.tensorflow.layers.made import MADE
from edward2.tensorflow.layers.neural_process import Attention
from edward2.tensorflow.layers.neural_process import NeuralProcess
from edward2.tensorflow.layers.noise import NCPCategoricalPerturb
from edward2.tensorflow.layers.noise import NCPNormalOutput
from edward2.tensorflow.layers.noise import NCPNormalPerturb
from edward2.tensorflow.layers.normalization import ActNorm
from edward2.tensorflow.layers.recurrent import LSTMCellFlipout
from edward2.tensorflow.layers.recurrent import LSTMCellReparameterization
from edward2.tensorflow.layers.stochastic_output import MixtureLogistic

from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

_allowed_symbols = [
    "ActNorm",
    "Attention",
    "BayesianLinearModel",
    "Conv2DFlipout",
    "Conv2DHierarchical",
    "Conv2DReparameterization",
    "Conv2DVariationalDropout",
    "DenseDVI",
    "DenseFlipout",
    "DenseHierarchical",
    "DenseReparameterization",
    "DenseVariationalDropout",
    "DiscreteAutoregressiveFlow",
    "DiscreteBipartiteFlow",
    "ExponentiatedQuadratic",
    "GaussianProcess",
    "LinearKernel",
    "LSTMCellFlipout",
    "LSTMCellReparameterization",
    "MADE",
    "MixtureLogistic",
    "NCPCategoricalPerturb",
    "NCPNormalOutput",
    "NCPNormalPerturb",
    "NeuralProcess",
    "Reverse",
    "SinkhornAutoregressiveFlow",
    "SparseGaussianProcess",
    "Zeros",
    "utils",
]

remove_undocumented(__name__, _allowed_symbols)
