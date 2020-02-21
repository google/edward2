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

# Lint as: python2, python3
"""Tests for metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward2 as ed
import numpy as np
import tensorflow.compat.v2 as tf


class ExpectedCalibrationErrorTest(tf.test.TestCase):

  def testOneClassFailure(self):
    with self.assertRaises(ValueError):
      ed.metrics.ExpectedCalibrationError(1)

  def testBinaryClassification(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array([0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68,
                              0.71, 0.81 + 0.85, 0])

    correct_ece = 0.
    bin_accs = np.array([0.] * num_bins)
    bin_confs = np.array([0.] * num_bins)
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = ed.metrics.ExpectedCalibrationError(
        2, num_bins, name='ECE', dtype=tf.float64)
    self.assertEqual(len(metric.variables), 3)

    metric.update_state(labels[:4], pred_probs[:4])
    ece1 = metric(labels[4:], pred_probs[4:])
    ece2 = metric.result()
    self.assertAllClose(ece1, ece2)
    self.assertAllClose(ece2, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def testBinaryClassificationKerasModel(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 0, 0, 2, 3, 1, 2, 0]
    bin_correct_sums = [0, 0, 0, 0, 0, 1, 2, 0, 2, 0]
    bin_prob_sums = [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71,
                     0.81 + 0.85, 0]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = ed.metrics.ExpectedCalibrationError(2, num_bins, name='ECE')
    self.assertEqual(len(metric.variables), 3)

    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: 1*x)])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[metric])
    outputs = model.predict(pred_probs)
    self.assertAllClose(pred_probs.reshape([n, 1]), outputs)
    _, ece = model.evaluate(pred_probs, labels)
    self.assertAllClose(ece, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def testMulticlassClassification(self):
    num_classes = 3
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 3, 0, 0, 2, 0, 0, 1]
    bin_correct_sums = [0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0, 0, 0, 0.32 + 0.37 + 0.39, 0, 0, 0.61 + 0.65, 0, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = ed.metrics.ExpectedCalibrationError(
        num_classes, num_bins, name='ECE', dtype=tf.float64)
    self.assertEqual(len(metric.variables), 3)

    metric.update_state(labels[:4], pred_probs[:4])
    ece1 = metric(labels[4:], pred_probs[4:])
    ece2 = metric.result()
    self.assertAllClose(ece1, ece2)
    self.assertAllClose(ece2, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
