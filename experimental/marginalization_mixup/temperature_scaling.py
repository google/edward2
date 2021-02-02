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

"""Temperature scaling on CIFAR-10 and CIFAR-100.

It takes a SavedModel, adds a temperature parameter to its predictions, and
minimizes negative log-likelihood with respect to the parameter by grid search.
"""

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
from experimental.marginalization_mixup import data_utils  # local file import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.baselines.cifar import utils
import uncertainty_metrics as um

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_bool('ensemble_then_calibrate', False,
                  'Whether to ensemble before or after scaling by temperature.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar',
                    'The directory where summaries are stored.')
flags.DEFINE_string('model_dir', '/tmp/cifar/model',
                    'The directory with SavedModel.')
flags.DEFINE_bool('corruptions', True, 'Whether to evaluate on corruptions.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_bool('save_predictions', False, 'Whether to save predictions.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving summaries at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  validation_input_fn = data_utils.load_input_fn(
      split=tfds.Split.VALIDATION,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      validation_set=True)
  clean_test_input_fn = data_utils.load_input_fn(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  test_datasets = {
      'validation': strategy.experimental_distribute_datasets_from_function(
          validation_input_fn),
      'clean': strategy.experimental_distribute_datasets_from_function(
          clean_test_input_fn),
  }
  if FLAGS.corruptions:
    if FLAGS.dataset == 'cifar10':
      load_c_dataset = utils.load_cifar10_c
    else:
      load_c_dataset = functools.partial(utils.load_cifar100_c,
                                         path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset = load_c_dataset(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

  ds_info = tfds.builder(FLAGS.dataset).info
  # TODO(ywenxu): Remove hard-coding validation images.
  steps_per_val = 2500 // (FLAGS.per_core_batch_size * FLAGS.num_cores)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size

  if FLAGS.use_bfloat16:
    policy = tf.python.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.python.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
    model = tf.python.keras.models.load_model(FLAGS.model_dir)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    # Compute grid search over [0.1, ..., 4.9, 5.0].
    temperatures = [x * 0.1 for x in range(1, 51)]
    temperature_metrics = []
    temperature_corrupt_metrics = []
    for _ in temperatures:
      metrics = {
          'val/negative_log_likelihood': tf.python.keras.metrics.Mean(),
          'val/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
          'val/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
          'test/negative_log_likelihood': tf.python.keras.metrics.Mean(),
          'test/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
          'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
      }
      temperature_metrics.append(metrics)
      corrupt_metrics = {}
      if FLAGS.corruptions:
        for intensity in range(1, max_intensity + 1):
          for corruption in corruption_types:
            dataset_name = '{0}_{1}'.format(corruption, intensity)
            corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
                tf.python.keras.metrics.Mean())
            corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
                tf.python.keras.metrics.SparseCategoricalAccuracy())
            corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
                um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
      temperature_corrupt_metrics.append(corrupt_metrics)

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)

      for (temperature,
           metrics,
           corrupt_metrics) in zip(temperatures,
                                   temperature_metrics,
                                   temperature_corrupt_metrics):
        tempered_logits = logits
        if not FLAGS.ensemble_then_calibrate:
          tempered_logits = logits / temperature
        probs = tf.nn.softmax(tempered_logits)
        per_probs = tf.split(probs,
                             num_or_size_splits=FLAGS.ensemble_size,
                             axis=0)
        probs = tf.reduce_mean(per_probs, axis=0)
        if FLAGS.ensemble_then_calibrate:
          probs = tf.nn.softmax(tf.math.log(probs) / temperature)
        negative_log_likelihood = tf.reduce_mean(
            tf.python.keras.losses.sparse_categorical_crossentropy(labels, probs))
        if dataset_name == 'validation':
          metrics['val/negative_log_likelihood'].update_state(
              negative_log_likelihood)
          metrics['val/accuracy'].update_state(labels, probs)
          metrics['val/ece'].update_state(labels, probs)
        elif dataset_name == 'clean':
          metrics['test/negative_log_likelihood'].update_state(
              negative_log_likelihood)
          metrics['test/accuracy'].update_state(labels, probs)
          metrics['test/ece'].update_state(labels, probs)
        else:
          corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
              negative_log_likelihood)
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
              labels, probs)
          corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
              labels, probs)
      return logits, labels

    return strategy.run(step_fn, args=(next(iterator),))

  start_time = time.time()
  for i, (dataset_name, test_dataset) in enumerate(test_datasets.items()):
    logging.info('Testing on dataset %s', dataset_name)
    test_iterator = iter(test_dataset)
    if dataset_name == 'validation':
      steps = steps_per_val
    else:
      steps = steps_per_eval
    full_logits = []
    full_labels = []
    for step in range(steps):
      logits, labels = test_step(test_iterator, dataset_name)
      full_logits.append(logits)
      full_labels.append(labels)

      current_step = i * steps_per_eval + (step + 1)
      max_steps = steps_per_eval * len(test_datasets)
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = ('{:.1%} completion: dataset {:d}/{:d}. {:.1f} steps/s. '
                 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                     current_step / max_steps,
                     i + 1,
                     len(test_datasets),
                     steps_per_sec,
                     eta_seconds / 60,
                     time_elapsed / 60))
      if step % 20 == 0:
        logging.info(message)

    if FLAGS.save_predictions:
      full_logits = tf.concat([
          tf.concat(strategy.experimental_local_results(logits), axis=0)
          for logits in full_logits], axis=0)
      full_labels = tf.cast(tf.concat([
          tf.concat(strategy.experimental_local_results(labels), axis=0)
          for labels in full_labels], axis=0), tf.int64)
      with tf.io.gfile.GFile(os.path.join(
          FLAGS.output_dir, f'{dataset_name}_logits.npy'), 'wb') as f:
        np.save(f, full_logits.numpy())
      with tf.io.gfile.GFile(os.path.join(
          FLAGS.output_dir, f'{dataset_name}_labels.npy'), 'wb') as f:
        np.save(f, full_labels.numpy())
    logging.info('Done with testing on %s', dataset_name)

  for i, metrics in enumerate(temperature_metrics):
    total_results = {name: metric.result() for name, metric in metrics.items()}
    if FLAGS.corruptions:
      corrupt_results = utils.aggregate_corrupt_metrics(
          temperature_corrupt_metrics[i],
          corruption_types,
          max_intensity,
          output_dir=FLAGS.output_dir)
      total_results.update(corrupt_results)
    logging.info('Temperature: %.2f, Test NLL: %.4f, Accuracy: %.2f%%',
                 temperatures[i],
                 total_results['test/negative_log_likelihood'],
                 total_results['test/accuracy'] * 100)
    # Use step counter as an alternative to temperature on x-axis. Can't do the
    # latter.
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=i)

  logging.info('Completed script')


if __name__ == '__main__':
  app.run(main)
