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

"""Utilities for CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def load_dataset(split, with_info=False):
  """Returns a tf.data.Dataset with <image, label> pairs.

  Args:
    split: tfds.Split.
    with_info: bool.

  Returns:
    Tuple of (tf.data.Dataset, tf.data.DatasetInfo) if with_info else only
    the dataset.
  """
  dataset, ds_info = tfds.load('cifar10',
                               split=split,
                               with_info=True,
                               batch_size=-1)
  image_shape = ds_info.features['image'].shape
  numpy_ds = tfds.as_numpy(dataset)
  numpy_images, numpy_labels = numpy_ds['image'], numpy_ds['label']
  if split == tfds.Split.TRAIN:
    numpy_images = numpy_images[:40000]
    numpy_labels = numpy_labels[:40000]
  dataset = tf.data.Dataset.from_tensor_slices((numpy_images, numpy_labels))

  def preprocess(image, label):
    """Image preprocessing function."""
    if split == tfds.Split.TRAIN:
      image = tf.image.random_flip_left_right(image)
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
      image = tf.image.random_crop(image, image_shape)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

  dataset = dataset.map(preprocess)
  if with_info:
    return dataset, ds_info
  return dataset


def make_lr_scheduler(init_lr):
  """Builds a keras LearningRateScheduler."""

  def schedule_fn(epoch):
    """Learning rate schedule function."""
    rate = init_lr
    if epoch > 180:
      rate *= 0.5e-3
    elif epoch > 160:
      rate *= 1e-3
    elif epoch > 120:
      rate *= 1e-2
    elif epoch > 80:
      rate *= 1e-1
    logging.info('Learning rate=%f for epoch=%d ', rate, epoch)
    return rate
  return tf.keras.callbacks.LearningRateScheduler(schedule_fn)
