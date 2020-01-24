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

"""Utilities for CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def load_dataset(split, with_info=False, data_augmentation=True):
  """Returns a tf.data.Dataset with <image, label> pairs.

  Args:
    split: tfds.Split.
    with_info: bool.
    data_augmentation: bool, if True perform simple data augmentation on the
      TRAIN split with random left/right flips and random cropping.  If False,
      do not perform any data augmentation.

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
  dataset = tf.data.Dataset.from_tensor_slices((numpy_images, numpy_labels))

  def preprocess(image, label):
    """Image preprocessing function."""
    if data_augmentation and split == tfds.Split.TRAIN:
      image = tf.image.random_flip_left_right(image)
      image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
      image = tf.image.random_crop(image, image_shape)

    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label

  dataset = dataset.map(preprocess)
  if with_info:
    return dataset, ds_info
  return dataset


def load_corrupted_test_dataset(batch_size,
                                name,
                                intensity,
                                use_bfloat16,
                                normalize=False):
  """Load a CIFAR-10-C dataset for testing."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  corruption = name + '_' + str(intensity)
  dataset = tfds.load(name='cifar10_corrupted/{}'.format(corruption),
                      split=tfds.Split.TEST,
                      as_supervised=True)

  def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465])
      std = tf.constant([0.2023, 0.1994, 0.2010])
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


# TODO(trandustin): Merge with load_dataset.
def load_distributed_dataset(split,
                             batch_size,
                             name,
                             use_bfloat16,
                             normalize=False,
                             with_info=False,
                             proportion=1.0):
  """Loads CIFAR dataset for training or testing.

  Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    with_info: bool.
    proportion: float, the proportion of dataset to be used.

  Returns:
    Tuple of (tf.data.Dataset, tf.data.DatasetInfo) if with_info else only
    the dataset.
  """
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  if proportion == 1.0:
    dataset, ds_info = tfds.load(name,
                                 split=split,
                                 with_info=True,
                                 as_supervised=True)
  else:
    name = '{}:3.*.*'.format(name)
    # TODO(ywenxu): consider the case where we have splits of train, val, test.
    if split == tfds.Split.TRAIN:
      split_str = 'train[:{}%]'.format(int(100 * proportion))
    else:
      split_str = 'test[:{}%]'.format(int(100 * proportion))
    dataset, ds_info = tfds.load(name,
                                 split=split_str,
                                 with_info=True,
                                 as_supervised=True)

  if split == tfds.Split.TRAIN:
    dataset_size = ds_info.splits['train'].num_examples
    dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

  image_shape = ds_info.features['image'].shape

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

  dataset = dataset.map(preprocess,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=split == tfds.Split.TRAIN)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
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


# TODO(trandustin): Merge with make_lr_scheduler.
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


# TODO(ghassen): disagreement and double_fault could be extended beyond pairs.
def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.reduce_mean(tf.cast(preds_1 != preds_2, tf.float32))


def double_fault(logits_1, logits_2, labels):
  """Double fault[1] is the number of examples both classifiers predict wrong.

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


def logit_kl_divergence(logits_1, logits_2):
  """Average KL divergence between logit distributions of two classifiers."""
  probs_1 = tf.nn.softmax(logits_1)
  probs_2 = tf.nn.softmax(logits_2)
  vals = kl_divergence(probs_1, probs_2)
  return tf.reduce_mean(vals)


def kl_divergence(p, q):
  """Generalized KL divergence [1] for unnormalized distributions.

  Args:
    p: tf.Tensor.
    q: tf.Tensor

  Returns:
    tf.Tensor of the Kullback-Leibler divergences per example.

  ## References

  [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative
  matrix factorization." Advances in neural information processing systems.
  2001.
  """
  return tf.reduce_sum(p * tf.math.log(p / q) - p + q, axis=-1)


def lp_distance(x, y, p=1):
  """l_p distance."""
  diffs_abs = tf.abs(x - y)
  summation = tf.reduce_sum(tf.math.pow(diffs_abs, p), axis=-1)
  return tf.reduce_mean(tf.math.pow(summation, 1./p), axis=-1)


def cosine_distance(x, y):
  """Cosine distance between vectors x and y."""
  x_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
  x_norm = tf.reshape(x_norm, (-1, 1))
  y_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1))
  y_norm = tf.reshape(y_norm, (-1, 1))
  normalized_x = x / x_norm
  normalized_y = y / y_norm
  return tf.reduce_mean(tf.reduce_sum(normalized_x * normalized_y, axis=-1))


# TODO(ghassen): we could extend this to take an arbitrary list of metric fns.
def average_pairwise_diversity(probs, labels, num_models):
  """Average pairwise distance computation across models."""
  if probs.shape[0] != num_models:
    raise ValueError('The number of models {0} does not match '
                     'the probs length {1}'.format(num_models, probs.shape[0]))

  pairwise_disagreement = []
  pairwise_double_fault = []
  pairwise_kl_divergence = []
  pairwise_cosine_distance = []
  for pair in list(itertools.combinations(range(num_models), 2)):
    probs_1 = probs[pair[0]]
    probs_2 = probs[pair[1]]
    pairwise_disagreement.append(disagreement(probs_1, probs_2))
    pairwise_double_fault.append(double_fault(probs_1, probs_2, labels))
    pairwise_kl_divergence.append(
        tf.reduce_mean(kl_divergence(probs_1, probs_2)))
    pairwise_cosine_distance.append(cosine_distance(probs_1, probs_2))

  # TODO(ghassen): we could also return max and min pairwise metrics.
  average_disagreement = tf.reduce_mean(tf.stack(pairwise_disagreement))
  average_double_fault = tf.reduce_mean(tf.stack(pairwise_double_fault))
  average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))
  average_cosine_distance = tf.reduce_mean(tf.stack(pairwise_cosine_distance))

  return {
      'disagreement': average_disagreement,
      'double_fault': average_double_fault,
      'average_kl': average_kl_divergence,
      'cosine_similarity': average_cosine_distance
  }
