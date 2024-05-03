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

"""Data utilities for CIFAR-10 and CIFAR-100."""

import functools
from experimental.marginalization_mixup import augment  # local file import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
tfd = tfp.distributions


def normalize_convert_image(input_image, dtype):
  input_image = tf.image.convert_image_dtype(input_image, dtype)
  mean = tf.constant([0.4914, 0.4822, 0.4465])
  std = tf.constant([0.2023, 0.1994, 0.2010])
  return (input_image - mean) / std


def load_input_fn(split,
                  batch_size,
                  name,
                  use_bfloat16,
                  normalize=True,
                  drop_remainder=True,
                  proportion=1.0,
                  validation_set=False,
                  aug_params=None):
  """Loads CIFAR dataset for training or testing.

  Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    proportion: float, the proportion of dataset to be used.
    validation_set: bool, whehter to split a validation set from training data.
    aug_params: dict, data augmentation hyper parameters.

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
  num_classes = ds_info.features['label'].num_classes
  if aug_params is None:
    aug_params = {}
  adaptive_mixup = aug_params.get('adaptive_mixup', False)
  random_augment = aug_params.get('random_augment', False)
  mixup_alpha = aug_params.get('mixup_alpha', 0)
  ensemble_size = aug_params.get('ensemble_size', 1)
  label_smoothing = aug_params.get('label_smoothing', 0.)
  forget_mixup = aug_params.get('forget_mixup', False)
  use_cutmix = aug_params.get('cutmix', False)

  if (adaptive_mixup or forget_mixup) and 'mixup_coeff' not in aug_params:
    # Hard target in the first epoch.
    aug_params['mixup_coeff'] = tf.ones([ensemble_size, num_classes])
  if mixup_alpha > 0 or label_smoothing > 0 or cutmix:
    onehot = True
  else:
    onehot = False

  def preprocess(image, label):
    """Image preprocessing function."""
    if split == tfds.Split.TRAIN:
      image = tf.image.resize_with_crop_or_pad(
          image, image_shape[0] + 4, image_shape[1] + 4)
      image = tf.image.random_crop(image, image_shape)
      image = tf.image.random_flip_left_right(image)

      # Only random augment for now.
      if random_augment:
        count = aug_params['aug_count']
        augmenter = augment.RandAugment()
        augmented = [augmenter.distort(image) for _ in range(count)]
        image = tf.stack(augmented)

    if split == tfds.Split.TRAIN and aug_params['augmix']:
      augmenter = augment.RandAugment()
      image = _augmix(image, aug_params, augmenter, dtype)
    elif normalize:
      image = normalize_convert_image(image, dtype)

    if split == tfds.Split.TRAIN and onehot:
      label = tf.cast(label, tf.int32)
      label = tf.one_hot(label, num_classes)
    else:
      label = tf.cast(label, dtype)
    return image, label

  def preprocess_with_idx(image, label, idx):
    """Image preprocessing function."""
    image, label = preprocess(image, label)
    return image, label, idx

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    if proportion == 1.0:
      if validation_set:
        if split == 'validation':
          dataset = tfds.load(name, split='train[95%:]', as_supervised=True)
        elif split == tfds.Split.TRAIN:
          dataset = tfds.load(name, split='train[:95%]', as_supervised=True)
        # split == tfds.Split.TEST case
        else:
          dataset = tfds.load(name, split=split, as_supervised=True)
      else:
        if forget_mixup and split == tfds.Split.TRAIN:
          full_data = tfds.load(name, split=split,
                                as_supervised=True, batch_size=-1)
          training_idx = tf.range(dataset_size)
          dataset = tf.data.Dataset.from_tensor_slices(
              (full_data[0], full_data[1], training_idx))
        else:
          dataset = tfds.load(name, split=split, as_supervised=True)
    else:
      new_name = '{}:3.*.*'.format(name)
      if split == tfds.Split.TRAIN:
        new_split = 'train[:{}%]'.format(int(100 * proportion))
      else:
        new_split = 'test[:{}%]'.format(int(100 * proportion))
      dataset = tfds.load(new_name, split=new_split, as_supervised=True)

    if split == tfds.Split.TRAIN:
      if forget_mixup:
        dataset = dataset.shuffle(buffer_size=dataset_size)
        dataset = dataset.map(preprocess_with_idx,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
      else:
        dataset = dataset.shuffle(buffer_size=dataset_size).repeat()
        dataset = dataset.map(preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    else:
      dataset = dataset.map(preprocess,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    if mixup_alpha > 0 and split == tfds.Split.TRAIN:
      if adaptive_mixup or forget_mixup:
        dataset = dataset.map(
            functools.partial(adaptive_mixup_aug, batch_size, aug_params),
            num_parallel_calls=8)
      else:
        dataset = dataset.map(
            functools.partial(mixup, batch_size, aug_params),
            num_parallel_calls=8)
    elif use_cutmix and split == tfds.Split.TRAIN:
      dataset = dataset.map(functools.partial(cutmix), num_parallel_calls=8)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


def augment_and_mix(image, depth, width, prob_coeff, augmenter, dtype):
  """Apply mixture of augmentations to image."""

  mix_weight = tf.squeeze(tfd.Beta([prob_coeff], [prob_coeff]).sample([1]))

  if width > 1:
    branch_weights = tf.squeeze(tfd.Dirichlet([prob_coeff] * width).sample([1]))
  else:
    branch_weights = tf.constant([1.])

  if depth < 0:
    depth = tf.random.uniform([width],
                              minval=1,
                              maxval=4,
                              dtype=tf.dtypes.int32)
  else:
    depth = tf.constant([depth] * width)

  mix = tf.cast(tf.zeros_like(image), tf.float32)
  for i in tf.range(width):
    branch_img = tf.identity(image)
    for _ in tf.range(depth[i]):
      branch_img = augmenter.distort(branch_img)
    branch_img = normalize_convert_image(branch_img, dtype)
    mix += branch_weights[i] * branch_img

  return mix_weight * mix + (
      1 - mix_weight) * normalize_convert_image(image, dtype)


def _augmix(image, params, augmenter, dtype):
  """Apply augmix augmentation to image."""
  depth = params['augmix_depth']
  width = params['augmix_width']
  prob_coeff = params['augmix_prob_coeff']
  count = params['aug_count']

  augmented = [
      augment_and_mix(image, depth, width, prob_coeff, augmenter, dtype)
      for _ in range(count)
  ]
  image = normalize_convert_image(image, dtype)
  return tf.stack([image] + augmented, 0)


def mixup(batch_size, aug_params, images, labels, idx=None):
  """Applies Mixup regularization to a batch of images and labels.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]
    idx: In the case of forget mixup, we need to remember the id of examples.

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  augmix = aug_params['augmix']
  alpha = aug_params['mixup_alpha']
  ensemble_size = aug_params['ensemble_size']

  if augmix:
    mix_weight = tfd.Beta(alpha, alpha).sample([batch_size, ensemble_size, 1])
  else:
    mix_weight = tfd.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  if augmix:
    images_mix_weight = tf.reshape(mix_weight,
                                   [batch_size, ensemble_size, 1, 1, 1])
  else:
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))

  if augmix:
    labels = tf.reshape(tf.tile(labels, [1, 4]), [batch_size, 4, -1])
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    labels_mix = tf.reshape(tf.transpose(
        labels_mix, [1, 0, 2]), [batch_size * 4, -1])
  else:
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)

  if idx is None:
    return images_mix, labels_mix
  else:
    return images_mix, labels_mix, idx


def adaptive_mixup_aug(batch_size, aug_params, images, labels, idx=None):
  """Applies Confidence Adjusted Mixup (CAMixup) regularization.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]
    idx: A batch of indices, either None or [batch_size]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  augmix = aug_params['augmix']
  ensemble_size = aug_params['ensemble_size']
  # Only run into the first branch if we are over the first epoch
  # after getting the forgetting counts
  if 'forget_counts_dir' in aug_params:
    threshold = aug_params['threshold']
    current_forget_path = aug_params['forget_counts_dir']
    with tf.io.gfile.GFile(current_forget_path, 'rb') as f:
      forget_counts = np.load(f)
    forget_counts = tf.convert_to_tensor(forget_counts)
    current_batch_forget_counts = tf.gather(forget_counts, idx, axis=-1)
    reverse_batch_forget_counts = current_batch_forget_counts[:, ::-1]
    total_forget_counts = (
        current_batch_forget_counts + reverse_batch_forget_counts)

    sorted_forget_counts = tf.sort(
        forget_counts, axis=-1, direction='DESCENDING')
    total_counts = tf.cast(tf.shape(sorted_forget_counts)[-1],
                           tf.constant(threshold).dtype)
    threshold_index = tf.cast(total_counts // threshold, tf.int32)
    # Reshape for boardcasting.
    threshold_value = tf.reshape(sorted_forget_counts[:, threshold_index],
                                 [ensemble_size, 1])
    mixup_coeff = tf.where(total_forget_counts > 2 * threshold_value,
                           aug_params['mixup_alpha'], 1.0)
    alpha = tf.reshape(mixup_coeff, [ensemble_size, batch_size])
  else:
    mixup_coeff = aug_params['mixup_coeff']
    scalar_labels = tf.argmax(labels, axis=1)
    alpha = tf.gather(mixup_coeff, scalar_labels, axis=-1)  # 4 x Batch_size

  # Need to filter out elements in alpha which equal to 0.
  greater_zero_indicator = tf.cast(alpha > 0, alpha.dtype)
  less_one_indicator = tf.cast(alpha < 1, alpha.dtype)
  valid_alpha_indicator = tf.cast(
      greater_zero_indicator * less_one_indicator, tf.bool)
  sampled_alpha = tf.where(valid_alpha_indicator, alpha, 0.1)
  mix_weight = tfd.Beta(sampled_alpha, sampled_alpha).sample()
  mix_weight = tf.where(valid_alpha_indicator, mix_weight, alpha)
  mix_weight = tf.reshape(mix_weight, [ensemble_size * batch_size, 1])
  mix_weight = tf.clip_by_value(mix_weight, 0, 1)
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  images_mix_weight = tf.reshape(mix_weight,
                                 [ensemble_size * batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  if augmix:
    images_shape = tf.shape(images)
    images = tf.reshape(tf.transpose(
        images, [1, 0, 2, 3, 4]), [-1, images_shape[2],
                                   images_shape[3], images_shape[4]])
  else:
    images = tf.tile(images, [ensemble_size, 1, 1, 1])
  labels = tf.tile(labels, [ensemble_size, 1])
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
  labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)

  if idx is None:
    return images_mix, labels_mix
  else:
    return images_mix, labels_mix, idx


def cutmix(images, labels):
  _, h, w, _ = images.shape.as_list()
  image1_proportion, mask = cutmix_padding(h, w)
  labels_mix = image1_proportion * labels + (
      1. - image1_proportion) * labels[::-1]
  images_mix = mask * images + (1. - mask) * images[::-1]
  return images_mix, labels_mix


def cutmix_padding(h, w):
  """Returns image mask for CutMix.

  Args:
    h: image height.
    w: image width.
  """
  r_x = tf.random.uniform([], 0, w, tf.int32)
  r_y = tf.random.uniform([], 0, h, tf.int32)

  # Beta dist in paper, but they used Beta(1,1) which is just uniform.
  image1_proportion = tf.random.uniform([])
  patch_length_ratio = tf.math.sqrt(1 - image1_proportion)
  r_w = tf.cast(patch_length_ratio * tf.cast(w, tf.float32), tf.int32)
  r_h = tf.cast(patch_length_ratio * tf.cast(h, tf.float32), tf.int32)
  bbx1 = tf.clip_by_value(tf.cast(r_x - r_w // 2, tf.int32), 0, w)
  bby1 = tf.clip_by_value(tf.cast(r_y - r_h // 2, tf.int32), 0, h)
  bbx2 = tf.clip_by_value(tf.cast(r_x + r_w // 2, tf.int32), 0, w)
  bby2 = tf.clip_by_value(tf.cast(r_y + r_h // 2, tf.int32), 0, h)

  # Create the binary mask.
  pad_left = bbx1
  pad_top = bby1
  pad_right = tf.maximum(w - bbx2, 0)
  pad_bottom = tf.maximum(h - bby2, 0)
  r_h = bby2 - bby1
  r_w = bbx2 - bbx1

  mask = tf.pad(
      tf.ones((r_h, r_w)),
      paddings=[[pad_top, pad_bottom], [pad_left, pad_right]],
      mode='CONSTANT',
      constant_values=0)
  mask.set_shape((h, w))
  return image1_proportion, mask[..., None]  # Add channel dim.
