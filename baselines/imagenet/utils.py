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

"""Utilities for ImageNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

IMAGE_SIZE = 224
CROP_PADDING = 32


# TODO(trandustin): Refactor similar to CIFAR code.
class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate, num_epochs,
               schedule):
    super(LearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.schedule = schedule

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self.schedule[0]
    # Scale learning rate schedule by total epochs at vanilla settings.
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self.schedule:
      start_epoch = (start_epoch * self.num_epochs) // 90
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
        'num_epochs': self.num_epochs,
        'schedule': self.schedule,
    }


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image_bytes: `Tensor` of binary image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image_bytes, bbox]):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        shape,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

    return image


def _at_least_x_are_equal(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are equal."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes):
  """Make a random crop of IMAGE_SIZE."""
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = distorted_bounding_box_crop(
      image_bytes,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4, 4. / 3.),
      area_range=(0.08, 1.0),
      max_attempts=10,
      scope=None)
  original_shape = tf.image.extract_jpeg_shape(image_bytes)
  bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

  image = tf.cond(
      bad,
      lambda: _decode_and_center_crop(image_bytes),
      lambda: tf.image.resize_bicubic([image],  # pylint: disable=g-long-lambda
                                      [IMAGE_SIZE, IMAGE_SIZE])[0])

  return image


def _decode_and_center_crop(image_bytes):
  """Crops to center of image with padding then scales IMAGE_SIZE."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize_bicubic([image], [IMAGE_SIZE, IMAGE_SIZE])[0]

  return image


def preprocess_for_train(image_bytes, use_bfloat16):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_random_crop(image_bytes)
  image = tf.image.random_flip_left_right(image)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_for_eval(image_bytes, use_bfloat16):
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    use_bfloat16: `bool` for whether to use bfloat16.

  Returns:
    A preprocessed image `Tensor`.
  """
  image = _decode_and_center_crop(image_bytes)
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  image = tf.image.convert_image_dtype(
      image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
  return image


def preprocess_image(image_bytes, is_training=False,
                     use_bfloat16=False):
  """Preprocesses the given image.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    is_training: `bool` for whether the preprocessing is for training.
    use_bfloat16: `bool` for whether to use bfloat16.

  Returns:
    A preprocessed image `Tensor`.
  """
  if is_training:
    return preprocess_for_train(image_bytes, use_bfloat16)
  else:
    return preprocess_for_eval(image_bytes, use_bfloat16)


class ImageNetInput(object):
  """Generates ImageNet input_fn for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Attributes:
    is_training: `bool` for whether the input is for training.
    data_dir: `str` for the directory of the training and validation data.
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    drop_remainder: `bool` for dropping the remainder when batching.
    batch_size: The global batch size to use.
    image_preprocessing_fn: Image preprocessing function.
  """

  def __init__(self,
               is_training,
               data_dir,
               batch_size,
               drop_remainder=True,
               use_bfloat16=False):
    self.image_preprocessing_fn = preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.drop_remainder = drop_remainder
    self.data_dir = data_dir
    self.batch_size = batch_size

  def dataset_parser(self, value):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = self.image_preprocessing_fn(
        image_bytes=image_bytes,
        is_training=self.is_training,
        use_bfloat16=self.use_bfloat16)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(tf.cast(
        tf.reshape(parsed['image/class/label'], shape=[1]), dtype=tf.int32) - 1,
                    dtype=tf.float32)

    return image, label

  def input_fn(self, ctx=None):
    """Input function which provides a single batch for train or eval.

    Args:
      ctx: Input context.

    Returns:
      A `tf.data.Dataset` object.
    """
    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if self.is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=self.is_training)

    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    if self.is_training:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        fetch_dataset, cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if self.is_training:
      dataset = dataset.shuffle(1024)

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            self.dataset_parser,
            batch_size=self.batch_size,
            num_parallel_batches=2,
            drop_remainder=self.drop_remainder))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if self.is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

    return dataset


def load_corrupted_test_dataset(batch_size,
                                name,
                                intensity,
                                drop_remainder=True,
                                use_bfloat16=False):
  """Loads an ImageNet-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  corruption = name + '_' + str(intensity)

  dataset = tfds.load(
      name='imagenet2012_corrupted/{}'.format(corruption),
      split=tfds.Split.VALIDATION,
      decoders={
          'image': tfds.decode.SkipDecoding(),
      },
      with_info=False,
      as_supervised=True)

  def preprocess(image, label):
    image = tf.reshape(image, shape=[])
    image = preprocess_for_eval(image, use_bfloat16)
    label = tf.cast(tf.cast(
        tf.reshape(label, shape=[1]), dtype=tf.int32) - 1,
                    dtype=dtype)
    return image, label

  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


def corrupt_test_input_fn(corruption_name,
                          corruption_intensity,
                          batch_size,
                          drop_remainder=True,
                          use_bfloat16=False):
  """Generates a distributed input_fn for ImageNet-C datasets."""
  def test_input_fn(ctx):
    """Sets up local (per-core) corrupted dataset batching."""
    dataset = load_corrupted_test_dataset(
        batch_size=batch_size,
        name=corruption_name,
        intensity=corruption_intensity,
        drop_remainder=drop_remainder,
        use_bfloat16=use_bfloat16)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset

  return test_input_fn


# TODO(ghassen,trandustin): Push this metadata upstream to TFDS.
def load_corrupted_test_info():
  """Loads information for ImageNet-C."""
  corruption_types = [
      'gaussian_noise',
      'shot_noise',
      'impulse_noise',
      'defocus_blur',
      'glass_blur',
      'motion_blur',
      'zoom_blur',
      'snow',
      'frost',
      'fog',
      'brightness',
      'contrast',
      'elastic_transform',
      'pixelate',
      'jpeg_compression',
  ]
  max_intensity = 5
  return corruption_types, max_intensity


def aggregate_corrupt_metrics(metrics, corruption_types, max_intensity):
  """Aggregates metrics across intensities and corruption types."""
  # TODO(trandustin): Decide whether to stick with mean or median; showing both
  # for now.
  results = {'test/nll_mean_corrupted': 0.,
             'test/error_mean_corrupted': 0.,
             'test/ece_mean_corrupted': 0.,
             'test/nll_median_corrupted': 0.,
             'test/error_median_corrupted': 0.,
             'test/ece_median_corrupted': 0.}
  for intensity in range(1, max_intensity + 1):
    ece = np.zeros(len(corruption_types))
    nll = np.zeros(len(corruption_types))
    acc = np.zeros(len(corruption_types))
    for i in range(len(corruption_types)):
      dataset_name = '{0}_{1}'.format(corruption_types[i], intensity)
      nll[i] = metrics['test/nll_{}'.format(dataset_name)].result()
      acc[i] = metrics['test/accuracy_{}'.format(dataset_name)].result()
      ece[i] = metrics['test/ece_{}'.format(dataset_name)].result()
    avg_nll = np.mean(nll)
    avg_error = 100 * (1 - 1. * np.mean(acc))
    avg_ece = np.mean(ece)
    median_nll = np.median(nll)
    median_error = 100 * (1. - 1. * np.median(acc))
    median_ece = np.median(ece)
    results['test/nll_mean_{}'.format(intensity)] = avg_nll
    results['test/error_mean_{}'.format(intensity)] = avg_error
    results['test/ece_mean_{}'.format(intensity)] = avg_ece
    results['test/nll_median_{}'.format(intensity)] = median_nll
    results['test/error_median_{}'.format(intensity)] = median_error
    results['test/ece_median_{}'.format(intensity)] = median_ece
    results['test/nll_mean_corrupted'] += avg_nll
    results['test/ece_mean_corrupted'] += avg_ece
    results['test/error_mean_corrupted'] += avg_error
    results['test/nll_median_corrupted'] += median_nll
    results['test/ece_median_corrupted'] += median_ece
    results['test/error_median_corrupted'] += median_error

  results['test/nll_mean_corrupted'] /= max_intensity
  results['test/error_mean_corrupted'] /= max_intensity
  results['test/ece_mean_corrupted'] /= max_intensity
  results['test/nll_median_corrupted'] /= max_intensity
  results['test/error_median_corrupted'] /= max_intensity
  results['test/ece_median_corrupted'] /= max_intensity
  return results


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
def average_pairwise_diversity(probs, num_models):
  """Average pairwise distance computation across models."""
  if probs.shape[0] != num_models:
    raise ValueError('The number of models {0} does not match '
                     'the probs length {1}'.format(num_models, probs.shape[0]))

  pairwise_disagreement = []
  pairwise_kl_divergence = []
  pairwise_cosine_distance = []
  for pair in list(itertools.combinations(range(num_models), 2)):
    probs_1 = probs[pair[0]]
    probs_2 = probs[pair[1]]
    pairwise_disagreement.append(disagreement(probs_1, probs_2))
    pairwise_kl_divergence.append(
        tf.reduce_mean(kl_divergence(probs_1, probs_2)))
    pairwise_cosine_distance.append(cosine_distance(probs_1, probs_2))

  # TODO(ghassen): we could also return max and min pairwise metrics.
  average_disagreement = tf.reduce_mean(tf.stack(pairwise_disagreement))
  average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))
  average_cosine_distance = tf.reduce_mean(tf.stack(pairwise_cosine_distance))

  return {
      'disagreement': average_disagreement,
      'average_kl': average_kl_divergence,
      'cosine_similarity': average_cosine_distance
  }
