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

"""Build a convolutional neural network."""

import edward2 as ed
import tensorflow as tf


def lenet5(input_shape, num_classes):
  """Builds LeNet5."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = tf.keras.layers.Conv2D(6,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation='relu')(inputs)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv1)
  conv2 = tf.keras.layers.Conv2D(16,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation='relu')(pool1)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
  conv3 = tf.keras.layers.Conv2D(120,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation=tf.nn.relu)(pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)
  logits = tf.keras.layers.Dense(num_classes)(dense1)
  outputs = tf.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(logits)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
