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

"""Layers."""

from edward2.tensorflow.layers import utils
from edward2.tensorflow.layers.bayesian_linear_model import BayesianLinearModel
from edward2.tensorflow.layers.convolutional import CondConv2D
from edward2.tensorflow.layers.convolutional import Conv1DBatchEnsemble
from edward2.tensorflow.layers.convolutional import Conv1DFlipout
from edward2.tensorflow.layers.convolutional import Conv1DRank1
from edward2.tensorflow.layers.convolutional import Conv1DReparameterization
from edward2.tensorflow.layers.convolutional import Conv2DBatchEnsemble
from edward2.tensorflow.layers.convolutional import Conv2DFlipout
from edward2.tensorflow.layers.convolutional import Conv2DHierarchical
from edward2.tensorflow.layers.convolutional import Conv2DRank1
from edward2.tensorflow.layers.convolutional import Conv2DReparameterization
from edward2.tensorflow.layers.convolutional import Conv2DVariationalDropout
from edward2.tensorflow.layers.convolutional import DepthwiseCondConv2D
from edward2.tensorflow.layers.convolutional import DepthwiseConv2DBatchEnsemble
from edward2.tensorflow.layers.dense import DenseBatchEnsemble
from edward2.tensorflow.layers.dense import DenseDVI
from edward2.tensorflow.layers.dense import DenseFlipout
from edward2.tensorflow.layers.dense import DenseHierarchical
from edward2.tensorflow.layers.dense import DenseRank1
from edward2.tensorflow.layers.dense import DenseReparameterization
from edward2.tensorflow.layers.dense import DenseVariationalDropout
from edward2.tensorflow.layers.discrete_flows import DiscreteAutoregressiveFlow
from edward2.tensorflow.layers.discrete_flows import DiscreteBipartiteFlow
from edward2.tensorflow.layers.discrete_flows import Reverse
from edward2.tensorflow.layers.discrete_flows import SinkhornAutoregressiveFlow
from edward2.tensorflow.layers.embeddings import EmbeddingReparameterization
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
from edward2.tensorflow.layers.normalization import ensemble_batchnorm
from edward2.tensorflow.layers.normalization import EnsembleSyncBatchNorm
from edward2.tensorflow.layers.normalization import SpectralNormalization
from edward2.tensorflow.layers.normalization import SpectralNormalizationConv2D
from edward2.tensorflow.layers.random_feature import LaplaceRandomFeatureCovariance
from edward2.tensorflow.layers.random_feature import RandomFeatureGaussianProcess
from edward2.tensorflow.layers.recurrent import LSTMCellFlipout
from edward2.tensorflow.layers.recurrent import LSTMCellRank1
from edward2.tensorflow.layers.recurrent import LSTMCellReparameterization
from edward2.tensorflow.layers.stochastic_output import MixtureLogistic

__all__ = [
    "ActNorm",
    "Attention",
    "BayesianLinearModel",
    "CondConv2D",
    "Conv1DBatchEnsemble",
    "Conv1DFlipout",
    "Conv1DRank1",
    "Conv1DReparameterization",
    "Conv2DBatchEnsemble",
    "Conv2DFlipout",
    "Conv2DHierarchical",
    "Conv2DRank1",
    "Conv2DReparameterization",
    "Conv2DVariationalDropout",
    "DenseBatchEnsemble",
    "DenseDVI",
    "DenseFlipout",
    "DenseHierarchical",
    "DenseReparameterization",
    "DenseRank1",
    "DenseVariationalDropout",
    "DepthwiseCondConv2D",
    "DepthwiseConv2DBatchEnsemble",
    "DiscreteAutoregressiveFlow",
    "DiscreteBipartiteFlow",
    "ExponentiatedQuadratic",
    "EmbeddingReparameterization",
    "EnsembleSyncBatchNorm",
    "GaussianProcess",
    "LaplaceRandomFeatureCovariance",
    "LinearKernel",
    "LSTMCellFlipout",
    "LSTMCellRank1",
    "LSTMCellReparameterization",
    "MADE",
    "MixtureLogistic",
    "NCPCategoricalPerturb",
    "NCPNormalOutput",
    "NCPNormalPerturb",
    "NeuralProcess",
    "RandomFeatureGaussianProcess",
    "Reverse",
    "SinkhornAutoregressiveFlow",
    "SparseGaussianProcess",
    "SpectralNormalization",
    "SpectralNormalizationConv2D",
    "Zeros",
    "ensemble_batchnorm",
    "utils",
]
