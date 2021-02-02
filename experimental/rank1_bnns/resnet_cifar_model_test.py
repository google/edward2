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

# Lint as: python3
"""Tests for Rank-1 ResNet-32x4."""
from absl.testing import parameterized
from experimental.rank1_bnns import resnet_cifar_model  # local file import
import tensorflow as tf


class ResnetCifarModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {'alpha_initializer': 'trainable_normal_fixed_stddev',
       'gamma_initializer': 'trainable_normal_fixed_stddev',
       'random_sign_init': 0.5,
       'ensemble_size': 1},
      {'alpha_initializer': 'trainable_deterministic',
       'gamma_initializer': 'trainable_deterministic',
       'random_sign_init': 0.5,
       'ensemble_size': 2},
      {'alpha_initializer': 'trainable_deterministic',
       'gamma_initializer': 'trainable_deterministic',
       'random_sign_init': -0.5,
       'ensemble_size': 2},
  )
  def testRank1ResNetV1(self,
                        alpha_initializer,
                        gamma_initializer,
                        random_sign_init,
                        ensemble_size):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 6
    input_shape = (32, 32, 2)  # TODO(dusenberrymw): (32, 32, 1) doesn't work...
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = resnet_cifar_model.rank1_resnet_v1(
        input_shape=input_shape,
        depth=8,
        num_classes=num_classes,
        width_multiplier=1,
        alpha_initializer=alpha_initializer,
        gamma_initializer=gamma_initializer,
        alpha_regularizer=None,
        gamma_regularizer=None,
        use_additive_perturbation=False,
        ensemble_size=ensemble_size,
        random_sign_init=-0.5,
        dropout_rate=0.)
    model.compile(
        'adam',
        loss=tf.python.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(dataset,
                        steps_per_epoch=dataset_size // batch_size,
                        epochs=2)

    loss_history = history.history['loss']
    self.assertAllGreaterEqual(loss_history, 0.)


if __name__ == '__main__':
  tf.test.main()
