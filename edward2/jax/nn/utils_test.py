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

"""Tests for utils."""
from absl.testing import absltest
from absl.testing import parameterized

import edward2.jax as ed

import jax
import jax.numpy as jnp

import numpy as np
import tensorflow as tf


class MeanFieldLogitsTest(parameterized.TestCase, tf.test.TestCase):

  def testMeanFieldLogitsLikelihood(self):
    """Tests if scaling is correct under different likelihood."""
    batch_size = 10
    num_classes = 12
    variance = 1.5
    mean_field_factor = 2.

    rng_key = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng_key, (batch_size, num_classes))
    covmat = jnp.ones(batch_size) * variance

    logits_logistic = ed.nn.utils.mean_field_logits(
        logits, covmat, mean_field_factor=mean_field_factor)
    logits_poisson = ed.nn.utils.mean_field_logits(
        logits,
        covmat,
        mean_field_factor=mean_field_factor,
        likelihood='poisson')

    self.assertAllClose(logits_logistic, logits / 2., atol=1e-4)
    self.assertAllClose(logits_poisson, logits * np.exp(1.5), atol=1e-4)

  def testMeanFieldLogitsTemperatureScaling(self):
    """Tests using mean_field_logits as temperature scaling method."""
    batch_size = 10
    num_classes = 12

    rng_key = jax.random.PRNGKey(0)
    logits = jax.random.normal(rng_key, (batch_size, num_classes))

    # Test if there's no change to logits when mean_field_factor < 0.
    logits_no_change = ed.nn.utils.mean_field_logits(
        logits, covmat=None, mean_field_factor=-1)

    # Test if mean_field_logits functions as a temperature scaling method when
    # mean_field_factor > 0, with temperature = sqrt(1. + mean_field_factor).
    logits_scale_by_two = ed.nn.utils.mean_field_logits(
        logits, covmat=None, mean_field_factor=3.)

    self.assertAllClose(logits_no_change, logits, atol=1e-4)
    self.assertAllClose(logits_scale_by_two, logits / 2., atol=1e-4)


if __name__ == '__main__':
  absltest.main()
