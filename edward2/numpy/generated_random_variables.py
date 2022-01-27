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

"""Automatically generated random variables."""

from edward2.trace import traceable
import scipy.stats

# Note a vanilla Edward2-like PPL in SciPy would introduce a RandomVariable
# abstraction: it wraps SciPy frozen distributions and calls `rvs` to associate
# the RandomVariable with a sampled value. SciPy distributions already enable
# parameters as input to `rvs`. Therefore instead of introducing a new
# abstraction, we just wrap `rvs`. This enables the same manipulations.
__all__ = []
_globals = globals()
for candidate_name in sorted(dir(scipy.stats)):
  candidate = getattr(scipy.stats, candidate_name)
  if isinstance(candidate, (scipy.stats._multivariate.multi_rv_generic,  # pylint: disable=protected-access
                            scipy.stats.rv_continuous,
                            scipy.stats.rv_discrete,
                            scipy.stats.rv_histogram)):
    candidate.rvs = traceable(candidate.rvs)
    _globals[candidate_name] = candidate
    __all__.append(candidate_name)

_HAS_DYNAMIC_ATTRIBUTES = True
