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

"""Tests for Keras-style regularizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.run_all_in_graph_and_eager_modes
class RegularizersTest(tf.test.TestCase):

  def testHalfCauchyKLDivergence(self):
    shape = (3,)
    regularizer = ed.regularizers.get('half_cauchy_kl_divergence')
    variational_posterior = ed.Independent(
        ed.LogNormal(loc=tf.zeros(shape), scale=1.).distribution,
        reinterpreted_batch_ndims=1)
    kl = regularizer(variational_posterior)
    kl_value = self.evaluate(kl)
    # KL uses a single-sample estimate, which is not necessarily >0. We only
    # check shape.
    self.assertEqual(kl_value.shape, ())


if __name__ == '__main__':
  tf.test.main()
