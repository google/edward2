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

"""Ensemble on ImageNet.

This script only performs evaluation, not training. We recommend training
ensembles by launching independent runs of `deterministic.py` over different
seeds. Set `output_dir` to the directory containing these checkpoints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import deterministic_model  # local file import
import utils  # local file import

import tensorflow.compat.v2 as tf

# TODO(trandustin): We inherit
# FLAGS.{dataset,per_core_batch_size,output_dir,seed} from deterministic. This
# is not intuitive, which suggests we need to either refactor to avoid importing
# from a binary or duplicate the model definition here.
flags.mark_flag_as_required('output_dir')
FLAGS = flags.FLAGS
NUM_CLASSES = 1000


def ensemble_negative_log_likelihood(labels, logits):
  """Negative log-likelihood for ensemble.

  For each datapoint (x,y), the ensemble's negative log-likelihood is:

  ```
  -log p(y|x) = -log sum_{m=1}^{ensemble_size} exp(log p(y|x,theta_m)) +
                log ensemble_size.
  ```

  Args:
    labels: tf.Tensor of shape [...].
    logits: tf.Tensor of shape [ensemble_size, ..., num_classes].

  Returns:
    tf.Tensor of shape [...].
  """
  labels = tf.cast(labels, tf.int32)
  logits = tf.convert_to_tensor(logits)
  ensemble_size = float(logits.shape[0])
  nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
      tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1]),
      logits)
  return -tf.reduce_logsumexp(-nll, axis=0) + tf.math.log(ensemble_size)


def gibbs_cross_entropy(labels, logits):
  """Average cross entropy for ensemble members (Gibbs cross entropy).

  For each datapoint (x,y), the ensemble's Gibbs cross entropy is:

  ```
  GCE = - (1/ensemble_size) sum_{m=1}^ensemble_size log p(y|x,theta_m).
  ```

  The Gibbs cross entropy approximates the average cross entropy of a single
  model drawn from the (Gibbs) ensemble.

  Args:
    labels: tf.Tensor of shape [...].
    logits: tf.Tensor of shape [ensemble_size, ..., num_classes].

  Returns:
    tf.Tensor of shape [...].
  """
  labels = tf.cast(labels, tf.int32)
  logits = tf.convert_to_tensor(logits)
  nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
      tf.broadcast_to(labels[tf.newaxis, ...], tf.shape(logits)[:-1]),
      logits)
  return tf.reduce_mean(nll, axis=0)


def main(argv):
  del argv  # unused arg
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)

  dataset_test = utils.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=False).input_fn()
  test_datasets = {'clean': dataset_test}

  model = deterministic_model.resnet50(input_shape=(224, 224, 3),
                                       num_classes=NUM_CLASSES)

  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())
  # Search for checkpoints from their index file; then remove the index suffix.
  ensemble_filenames = tf.io.gfile.glob(os.path.join(FLAGS.output_dir,
                                                     '**/*.index'))
  ensemble_filenames = [filename[:-6] for filename in ensemble_filenames]
  ensemble_size = len(ensemble_filenames)
  logging.info('Ensemble size: %s', ensemble_size)
  logging.info('Ensemble number of weights: %s',
               ensemble_size * model.count_params())
  logging.info('Ensemble filenames: %s', str(ensemble_filenames))
  checkpoint = tf.train.Checkpoint(model=model)

  # Collect the logits output for each ensemble member and test data
  # point. We also collect the labels.

  logits_test = {'clean': []}
  labels_test = {'clean': []}
  corruption_types, max_intensity = utils.load_corrupted_test_info()
  for name in corruption_types:
    for intensity in range(1, max_intensity + 1):
      dataset_name = '{0}_{1}'.format(name, intensity)
      logits_test[dataset_name] = []
      labels_test[dataset_name] = []

      test_datasets[dataset_name] = utils.load_corrupted_test_dataset(
          name=name,
          intensity=intensity,
          batch_size=FLAGS.per_core_batch_size,
          drop_remainder=True,
          use_bfloat16=False)

  for m, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename)
    logging.info('Working on test data for ensemble member %s', m)
    for name, test_dataset in test_datasets.items():
      logits = []
      for features, labels in test_dataset:
        logits.append(model(features, training=False))
        if m == 0:
          labels_test[name].append(labels)

      logits = tf.concat(logits, axis=0)
      logits_test[name].append(logits)
      if m == 0:
        labels_test[name] = tf.concat(labels_test[name], axis=0)
      logging.info('Finished testing on %s', format(name))

  metrics = {
      'test/ece': ed.metrics.ExpectedCalibrationError(num_classes=NUM_CLASSES,
                                                      num_bins=15)
  }
  corrupt_metrics = {}
  for name in test_datasets:
    corrupt_metrics['test/ece_{}'.format(
        name)] = ed.metrics.ExpectedCalibrationError(
            num_classes=NUM_CLASSES, num_bins=15)
    corrupt_metrics['test/nll_{}'.format(name)] = tf.keras.metrics.Mean()
    corrupt_metrics['test/accuracy_{}'.format(name)] = tf.keras.metrics.Mean()

  for name, test_dataset in test_datasets.items():
    labels = labels_test[name]
    logits = logits_test[name]
    nll_test = ensemble_negative_log_likelihood(labels, logits)
    gibbs_ce_test = gibbs_cross_entropy(labels_test[name], logits_test[name])
    labels = tf.cast(labels, tf.int32)
    logits = tf.convert_to_tensor(logits)
    per_probs = tf.nn.softmax(logits)
    probs = tf.reduce_mean(per_probs, axis=0)
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(labels, probs)
    if name == 'clean':
      metrics['test/negative_log_likelihood'] = tf.reduce_mean(nll_test)
      metrics['test/gibbs_cross_entropy'] = tf.reduce_mean(gibbs_ce_test)
      metrics['test/accuracy'] = tf.reduce_mean(accuracy)
      metrics['test/ece'].update_state(labels, probs)
    else:
      corrupt_metrics['test/nll_{}'.format(name)].update_state(
          tf.reduce_mean(nll_test))
      corrupt_metrics['test/accuracy_{}'.format(name)].update_state(
          tf.reduce_mean(accuracy))
      corrupt_metrics['test/ece_{}'.format(name)].update_state(labels, probs)

  corrupt_results = {}
  corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                    corruption_types,
                                                    max_intensity)
  metrics['test/ece'] = metrics['test/ece'].result()
  total_results = {name: metric for name, metric in metrics.items()}
  total_results.update(corrupt_results)
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
