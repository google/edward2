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

"""Tests for ed.losses."""

from absl.testing import parameterized
import edward2 as ed
import numpy as np
import tensorflow as tf


def test_cases_uce():
  sparsities = [True, False]
  nums_classes = [2, 6]
  entropy_regs = [1e-5, 1e-2]
  cases = []
  for sparse in sparsities:
    for num_classes in nums_classes:
      for entropy_reg in entropy_regs:
        cases.append({
            'testcase_name': f'_uce_loss_{sparse}_{num_classes}_{entropy_reg}',
            'sparse': sparse,
            'num_classes': num_classes,
            'entropy_reg': entropy_reg,
        })
  return parameterized.named_parameters(*cases)


class LossesTest(tf.test.TestCase, parameterized.TestCase):

  def _generate_data(self, sparse, num_classes):
    labels = np.random.randint(low=0, high=num_classes, size=(16,))
    if not sparse:
      labels = np.eye(num_classes)[labels]
    alphas = np.random.random(size=(16, num_classes))
    return (tf.convert_to_tensor(alphas, dtype=tf.float32),
            tf.convert_to_tensor(labels, dtype=tf.float32))

  @test_cases_uce()
  def test_uce_loss(self, sparse, num_classes, entropy_reg):
    alphas, labels = self._generate_data(sparse, num_classes)
    loss_fn = ed.losses.uce_loss(entropy_reg=entropy_reg,
                                 sparse=sparse,
                                 num_classes=num_classes)
    loss_value = loss_fn(labels, alphas)
    self.assertIsNotNone(loss_value)

  def test_uce_value(self):
    labels = np.random.randint(low=0, high=5, size=(16,))
    alphas = np.eye(5)[labels].astype(np.float32) + 1.
    loss_fn = ed.losses.uce_loss(sparse=True, entropy_reg=1e-6, num_classes=5)
    loss_value = loss_fn(labels, alphas)
    self.assertAllClose(loss_value, 1.2833369, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
