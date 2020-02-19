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

"""Compute metrics of an ensemble model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy


def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def brier_score(y, p):
  """Compute the Brier score.

  Brier Score: see
  https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf,
  page 363, Example 1

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p: numpy array, size (?, num_classes)
       containing the output predicted probabilities
  Returns:
    bs: Brier score.
  """
  return np.mean(np.power(p - y, 2))


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.

  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins

  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce


def ensemble_metrics(x,
                     y,
                     neural_net,
                     log_likelihood_fn,
                     n_samples=10,
                     weight_files=None):
  """Evaluate metrics of a Bayesian ensemble.

  Args:
    x: numpy array of inputs
    y: numpy array of labels
    neural_net: keras model
    log_likelihood_fn: keras function of log likelihood. For classification
      tasks, log_likelihood_fn(...)[1] should return the logits
    n_samples: number of Monte Carlo samples to draw. The samples are evenly
      divided aross multiple weight sets.
    weight_files: to draw samples from multiple weight sets, specify a list of
      weight files to load. These files must have been generated through
      keras's model.save_weights(...).

  Returns:
    metrics_dict: dictionary containing the metrics
  """

  # in classification, log_likelihood_fn[1] returns the logits,
  # which has shape > 1 while in regression it returns the error.
  classification = (
      log_likelihood_fn.outputs[1].shape.as_list()[1] > 1
  )
  if weight_files is None:
    ensemble_logprobs = [log_likelihood_fn([x, y])[0] for _ in range(n_samples)]
    metric_values = [neural_net.evaluate(x, y, verbose=0)
                     for _ in range(n_samples)]
    if classification:
      ensemble_logits = [log_likelihood_fn([x, y])[1] for _ in range(n_samples)]
    else:
      ensemble_error = [log_likelihood_fn([x, y])[1] for _ in range(n_samples)]
  else:
    ensemble_logprobs = []
    metric_values = []
    ensemble_logits = []
    ensemble_error = []
    for filename in weight_files:
      neural_net.load_weights(filename)
      ensemble_logprobs.extend([
          log_likelihood_fn([x, y])[0]
          for _ in range(n_samples // len(weight_files))
      ])
      if classification:
        ensemble_logits.extend([
            log_likelihood_fn([x, y])[1]
            for _ in range(n_samples // len(weight_files))
        ])
      else:
        ensemble_error.extend([
            log_likelihood_fn([x, y])[1]
            for _ in range(n_samples // len(weight_files))
        ])
      metric_values.extend([
          neural_net.evaluate(x, y, verbose=0)
          for _ in range(n_samples // len(weight_files))
      ])

  metric_values = np.mean(np.array(metric_values), axis=0)
  results = {}
  for m, name in zip(metric_values, neural_net.metrics_names):
    results[name] = m

  ensemble_logprobs = np.array(ensemble_logprobs)
  bayesian_mll = np.mean(
      scipy.special.logsumexp(
          np.sum(ensemble_logprobs, axis=2)
          if len(ensemble_logprobs.shape) > 2 else ensemble_logprobs,
          b=1. / ensemble_logprobs.shape[0],
          axis=0),
      axis=0)
  results['bayesian_mll'] = bayesian_mll

  if classification:
    ensemble_logits = np.array(ensemble_logits)
    probs = np.mean(scipy.special.softmax(ensemble_logits, axis=2), axis=0)
    class_pred = np.argmax(probs, axis=1)
    bayesian_accuracy = np.mean(np.equal(y, class_pred))
    results['bayesian_accuracy'] = bayesian_accuracy
    results['ece'], results['mce'] = calibration(
        one_hot(y, probs.shape[1]), probs)
    results['brier_score'] = brier_score(one_hot(y, probs.shape[1]), probs)
  else:
    ensemble_error = np.stack([np.array(l) for l in ensemble_error])
    results['bayesian_mse'] = np.mean(
        np.square(np.mean(ensemble_error, axis=0)))
  return results
