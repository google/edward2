# coding=utf-8
# Copyright 2020 The Edward2 Authors.
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

"""Utilities for CIFAR-10 and CIFAR-100."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def load_cifar100_c_input_fn(corruption_name,
                             corruption_intensity,
                             batch_size,
                             use_bfloat16,
                             path,
                             normalize=True):
  """Loads CIFAR-100-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  filename = path + '{0}-{1}.tfrecords'.format(corruption_name,
                                               corruption_intensity)
  def preprocess(serialized_example):
    """Preprocess a serialized example for CIFAR100-C."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.cast(tf.reshape(image, [32, 32, 3]), dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image = image / 255  # to convert into the [0, 1) range
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465])
      std = tf.constant([0.2023, 0.1994, 0.2010])
      image = (image - mean) / std
    else:
      # Normalize per-image using mean/stddev computed across pixels.
      image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


def load_cifar10_c_input_fn(corruption_name,
                            corruption_intensity,
                            batch_size,
                            use_bfloat16,
                            normalize=True):
  """Loads CIFAR-10-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  corruption = corruption_name + '_' + str(corruption_intensity)
  def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465])
      std = tf.constant([0.2023, 0.1994, 0.2010])
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tfds.load(name='cifar10_corrupted/{}'.format(corruption),
                        split=tfds.Split.TEST,
                        as_supervised=True)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


# TODO(ghassen,trandustin): Push this metadata upstream to TFDS.
def load_corrupted_test_info(dataset):
  """Loads information for CIFAR-10-C."""
  if dataset == 'cifar10':
    corruption_types = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'frosted_glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic',
        'pixelate',
        'jpeg_compression',
    ]
  else:
    corruption_types = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'glass_blur',  # Called frosted_glass_blur in CIFAR-10.
        'gaussian_blur',
        'gaussian_noise',
        'impulse_noise',
        'jpeg_compression',
        'pixelate',
        'saturate',
        'shot_noise',
        'spatter',
        'speckle_noise',  # Does not exist for CIFAR-10.
        'zoom_blur',
    ]
  max_intensity = 5
  return corruption_types, max_intensity


def load_input_fn(split,
                  batch_size,
                  name,
                  use_bfloat16,
                  normalize=True,
                  drop_remainder=True,
                  proportion=1.0):
  """Loads CIFAR dataset for training or testing.

  Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    proportion: float, the proportion of dataset to be used.

  Returns:
    Input function which returns a locally-sharded dataset batch.
  """
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  ds_info = tfds.builder(name).info
  image_shape = ds_info.features['image'].shape
  dataset_size = ds_info.splits['train'].num_examples

  def preprocess(image, label):
    """Image preprocessing function."""
    if split == tfds.Split.TRAIN:
      image = tf.image.resize_with_crop_or_pad(
          image, image_shape[0] + 4, image_shape[1] + 4)
      image = tf.image.random_crop(image, image_shape)
      image = tf.image.random_flip_left_right(image)

    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465])
      std = tf.constant([0.2023, 0.1994, 0.2010])
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    if proportion == 1.0:
      dataset = tfds.load(name, split=split, as_supervised=True)
    else:
      new_name = '{}:3.*.*'.format(name)
      if split == tfds.Split.TRAIN:
        new_split = 'train[:{}%]'.format(int(100 * proportion))
      else:
        new_split = 'test[:{}%]'.format(int(100 * proportion))
      dataset = tfds.load(new_name, split=new_split, as_supervised=True)
    if split == tfds.Split.TRAIN:
      dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               steps_per_epoch,
               initial_learning_rate,
               decay_ratio,
               decay_epochs,
               warmup_epochs):
    super(LearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    learning_rate = self.initial_learning_rate
    if self.warmup_epochs >= 1:
      learning_rate *= lr_epoch / self.warmup_epochs
    decay_epochs = [self.warmup_epochs] + self.decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
      learning_rate = tf.where(
          lr_epoch >= start_epoch,
          self.initial_learning_rate * self.decay_ratio**index,
          learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
    }


def aggregate_corrupt_metrics(metrics,
                              corruption_types,
                              max_intensity,
                              fine_metrics=False):
  """Aggregates metrics across intensities and corruption types."""
  results = {'test/nll_mean_corrupted': 0.,
             'test/accuracy_mean_corrupted': 0.,
             'test/ece_mean_corrupted': 0.,
             'test/brier_mean_corrupted': 0.}
  for intensity in range(1, max_intensity + 1):
    ece = np.zeros(len(corruption_types))
    nll = np.zeros(len(corruption_types))
    acc = np.zeros(len(corruption_types))
    bri = np.zeros(len(corruption_types))
    for i in range(len(corruption_types)):
      dataset_name = '{0}_{1}'.format(corruption_types[i], intensity)
      nll[i] = metrics['test/nll_{}'.format(dataset_name)].result()
      acc[i] = metrics['test/accuracy_{}'.format(dataset_name)].result()
      ece[i] = metrics['test/ece_{}'.format(dataset_name)].result()
      bri[i] = metrics['test/brier_{}'.format(dataset_name)].result()
      if fine_metrics:
        results['test/nll_{}'.format(dataset_name)] = nll[i]
        results['test/accuracy_{}'.format(dataset_name)] = acc[i]
        results['test/ece_{}'.format(dataset_name)] = ece[i]
        results['test/brier_{}'.format(dataset_name)] = bri[i]
    avg_nll = np.mean(nll)
    avg_acc = np.mean(acc)
    avg_ece = np.mean(ece)
    avg_bri = np.mean(bri)
    results['test/nll_mean_{}'.format(intensity)] = avg_nll
    results['test/accuracy_mean_{}'.format(intensity)] = avg_acc
    results['test/ece_mean_{}'.format(intensity)] = avg_ece
    results['test/brier_mean_corrupted{}'.format(intensity)] = avg_bri
    results['test/nll_mean_corrupted'] += avg_nll
    results['test/accuracy_mean_corrupted'] += avg_acc
    results['test/ece_mean_corrupted'] += avg_ece
    results['test/brier_mean_corrupted'] += avg_bri

  results['test/nll_mean_corrupted'] /= max_intensity
  results['test/accuracy_mean_corrupted'] /= max_intensity
  results['test/ece_mean_corrupted'] /= max_intensity
  results['test/brier_mean_corrupted'] /= max_intensity
  return results


def double_fault(logits_1, logits_2, labels):
  """Double fault [1] is the number of examples both classifiers predict wrong.

  Args:
    logits_1: tf.Tensor.
    logits_2: tf.Tensor.
    labels: tf.Tensor.

  Returns:
    Scalar double-fault diversity metric.

  ## References

  [1] Kuncheva, Ludmila I., and Christopher J. Whitaker. "Measures of diversity
      in classifier ensembles and their relationship with the ensemble
      accuracy." Machine learning 51.2 (2003): 181-207.
  """
  preds_1 = tf.cast(tf.argmax(logits_1, axis=-1), labels.dtype)
  preds_2 = tf.cast(tf.argmax(logits_2, axis=-1), labels.dtype)

  fault_1_idx = tf.squeeze(tf.where(preds_1 != labels))
  fault_1_idx = tf.cast(fault_1_idx, tf.int32)

  preds_2_at_idx = tf.gather(preds_2, fault_1_idx)
  labels_at_idx = tf.gather(labels, fault_1_idx)

  double_faults = preds_2_at_idx != labels_at_idx
  double_faults = tf.cast(double_faults, tf.float32)
  return tf.reduce_mean(double_faults)
