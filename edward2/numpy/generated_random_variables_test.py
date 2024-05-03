# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Tests for generated random variables."""

from absl.testing import absltest
import edward2.numpy as ed
import scipy.stats


class GeneratedRandomVariablesTest(absltest.TestCase):

  def testBernoulli(self):
    self.assertEqual(ed.bernoulli.__doc__, scipy.stats.bernoulli.__doc__)
    self.assertEqual(ed.bernoulli.logpmf(0, p=0.2),
                     scipy.stats.bernoulli.logpmf(0, p=0.2))

if __name__ == "__main__":
  absltest.main()
