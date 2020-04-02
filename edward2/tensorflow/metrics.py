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
"""Metrics (Keras-style)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import tensorflow.compat.v2 as tf


class ExpectedCalibrationError(tf.keras.metrics.Metric):
  """Expected Calibration Error.

  Expected calibration error (Guo et al., 2017, Naeini et al., 2015) is a scalar
  measure of calibration for probabilistic models. Calibration is defined as the
  level to which the accuracy over a set of predicted decisions and true
  outcomes associated with a given predicted probability level matches the
  predicted probability. A perfectly calibrated model would be correct `p`% of
  the time for all examples for which the predicted probability was `p`%, over
  all values of `p`.

  This metric can be computed as follows. First, cut up the probability space
  interval [0, 1] into some number of bins. Then, for each example, store the
  predicted class (based on a threshold of 0.5 in the binary case and the max
  probability in the multiclass case), the predicted probability corresponding
  to the predicted class, and the true label into the corresponding bin based on
  the predicted probability. Then, for each bin, compute the average predicted
  probability ("confidence"), the accuracy of the predicted classes, and the
  absolute difference between the confidence and the accuracy ("calibration
  error"). Expected calibration error can then be computed as a weighted average
  calibration error over all bins, weighted based on the number of examples per
  bin.

  Perfect calibration under this setup is when, for all bins, the average
  predicted probability matches the accuracy, and thus the expected calibration
  error equals zero. In the limit as the number of bins goes to infinity, the
  predicted probability would be equal to the accuracy for all possible
  probabilities.

  References:
    1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On Calibration of Modern
       Neural Networks. in International Conference on Machine Learning (ICML)
       cs.LG, (Cornell University Library, 2017).
    2. Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated
       Probabilities Using Bayesian Binning. Proc Conf AAAI Artif Intell 2015,
       2901-2907 (2015).
  """

  _setattr_tracking = False  # Automatic tracking breaks some unit tests

  def __init__(self, num_classes, num_bins=15, name=None, dtype=None):
    """Constructs an expected calibration error metric.

    Args:
      num_classes: Total number of classes.
      num_bins: Number of bins to maintain over the interval [0, 1].
      name: Name of this metric.
      dtype: Data type.
    """
    super(ExpectedCalibrationError, self).__init__(name, dtype)
    if num_classes < 2:
      raise ValueError(
          'Num classes must be >= 2. Given: {}.'.format(num_classes))
    self.num_classes = num_classes
    self.num_bins = num_bins

    self.correct_sums = self.add_weight(
        'correct_sums', shape=(num_bins,), initializer=tf.zeros_initializer)
    self.prob_sums = self.add_weight(
        'prob_sums', shape=(num_bins,), initializer=tf.zeros_initializer)
    self.counts = self.add_weight(
        'counts', shape=(num_bins,), initializer=tf.zeros_initializer)

  def update_state(self, labels, probabilities, **kwargs):
    """Updates this metric.

    Args:
      labels: Tensor of shape (N,) of class labels, one per example.
      probabilities: Tensor of shape (N,) or (N, k) of normalized probabilities
        associated with the True class in the binary case or with each of k
        classes in the multiclass case.
      **kwargs: Other potential keywords, which will be ignored by this method.
    """
    del kwargs  # unused
    labels = tf.squeeze(tf.convert_to_tensor(labels))
    probabilities = tf.convert_to_tensor(probabilities, self.dtype)

    if self.num_classes == 2:
      # Explicitly ensure probs have shape [n, 2] instead of [n, 1] or [n,].
      n = tf.shape(probabilities)[0]
      k = tf.size(probabilities) // n
      probabilities = tf.reshape(probabilities, [n, k])
      probabilities = tf.cond(
          k < 2, lambda: tf.concat([1. - probabilities, probabilities], axis=1),
          lambda: probabilities)

    pred_labels = tf.argmax(probabilities, axis=1)
    pred_probs = tf.reduce_max(probabilities, axis=1)
    correct_preds = tf.equal(pred_labels, tf.cast(labels, pred_labels.dtype))
    correct_preds = tf.cast(correct_preds, self.dtype)

    bin_indices = tf.histogram_fixed_width_bins(
        pred_probs, tf.constant([0., 1.], self.dtype), nbins=self.num_bins)
    batch_correct_sums = tf.math.unsorted_segment_sum(
        data=tf.cast(correct_preds, self.dtype),
        segment_ids=bin_indices,
        num_segments=self.num_bins)
    batch_prob_sums = tf.math.unsorted_segment_sum(data=pred_probs,
                                                   segment_ids=bin_indices,
                                                   num_segments=self.num_bins)
    batch_counts = tf.math.unsorted_segment_sum(data=tf.ones_like(bin_indices),
                                                segment_ids=bin_indices,
                                                num_segments=self.num_bins)
    batch_counts = tf.cast(batch_counts, self.dtype)
    self.correct_sums.assign_add(batch_correct_sums)
    self.prob_sums.assign_add(batch_prob_sums)
    self.counts.assign_add(batch_counts)

  def result(self):
    """Computes the expected calibration error."""
    non_empty = tf.math.not_equal(self.counts, 0)
    correct_sums = tf.boolean_mask(self.correct_sums, non_empty)
    prob_sums = tf.boolean_mask(self.prob_sums, non_empty)
    counts = tf.boolean_mask(self.counts, non_empty)
    accs = correct_sums / counts
    confs = prob_sums / counts
    total_count = tf.reduce_sum(counts)
    return tf.reduce_sum(counts / total_count * tf.abs(accs - confs))

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    tf.keras.backend.batch_set_value([(v, [0.,]*self.num_bins) for v in
                                      self.variables])


# TODO(ghassen): disagreement and double_fault could be extended beyond pairs.
def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.reduce_mean(tf.cast(preds_1 != preds_2, tf.float32))


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
def average_pairwise_diversity(probs, num_models, error=None):
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
  if error is not None:
    average_disagreement /= (error + tf.keras.backend.epsilon())
  average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))
  average_cosine_distance = tf.reduce_mean(tf.stack(pairwise_cosine_distance))

  return {
      'disagreement': average_disagreement,
      'average_kl': average_kl_divergence,
      'cosine_similarity': average_cosine_distance
  }
