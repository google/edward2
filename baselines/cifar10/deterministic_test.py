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

"""Tests for deterministic ResNet-20."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deterministic  # local file import
import tensorflow.compat.v2 as tf


class DeterministicTest(tf.test.TestCase):

  def testResNetV1(self):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 5
    input_shape = (32, 32, 3)
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    labels = tf.random.uniform((dataset_size,),
                               minval=0,
                               maxval=num_classes,
                               dtype=tf.int32)
    features, labels = self.evaluate([features, labels])
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = deterministic.resnet_v1(
        input_shape=input_shape,
        depth=8,
        num_classes=num_classes,
        batch_norm=False)
    def negative_log_likelihood(y_true, y_pred):
      y_true = tf.squeeze(y_true)
      return -y_pred.distribution.log_prob(y_true)
    model.compile(tf.keras.optimizers.Adam(),
                  loss=negative_log_likelihood)
    history = model.fit(dataset,
                        steps_per_epoch=dataset_size // batch_size,
                        epochs=1)

    # Check loss is non-negative, i.e., not NaN.
    self.assertAllGreaterEqual(history.history['loss'], 0.)


if __name__ == '__main__':
  tf.test.main()
