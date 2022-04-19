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

"""Implementation of spectral normalization for Dense and Conv2D layers.

## References:

[1] Yuichi Yoshida, Takeru Miyato. Spectral Norm Regularization for Improving
    the Generalizability of Deep Learning.
    _arXiv preprint arXiv:1705.10941_, 2017. https://arxiv.org/abs/1705.10941

[2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida.
    Spectral normalization for generative adversarial networks.
    In _International Conference on Learning Representations_, 2018.

[3] Henry Gouk, Eibe Frank, Bernhard Pfahringer, Michael Cree.
    Regularisation of neural networks by enforcing lipschitz continuity.
    _arXiv preprint arXiv:1804.04368_, 2018. https://arxiv.org/abs/1804.04368
"""
import warnings
import edward2 as ed


def SpectralNormalization(*args, **kwargs):
  warnings.warn(
      'Please use `edward2.layers.SpectralNormalization` instead.',
      category=DeprecationWarning,
      stacklevel=2)
  return ed.layers.SpectralNormalization(*args, **kwargs)


def SpectralNormalizationConv2D(*args, **kwargs):
  warnings.warn(
      'Please use `edward2.layers.SpectralNormalizationConv2D` instead.',
      category=DeprecationWarning,
      stacklevel=2)
  return ed.layers.SpectralNormalizationConv2D(*args, **kwargs)
