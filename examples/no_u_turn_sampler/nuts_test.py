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

"""Tests of the No U-Turn Sampler.

The majority of the tests are based on visually checking plots. For now,
we only test that the plots have no runtime errors (that is, they type-check).
"""

import os
from absl.testing import parameterized
import no_u_turn_sampler  # local file import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


def plot_with_expectation(samples, dist, model_dir, suffix):
  """Comparison histogram of samples and a line for the expected density."""
  _, bins, _ = plt.hist(samples, bins=30, label="observed")
  bin_width = bins[1] - bins[0]
  xs = np.linspace(bins[0], bins[-1], 500)
  pdfs = dist.pdf(xs) * bin_width * len(samples)
  plt.plot(xs, pdfs, label="analytic")
  plt.legend()
  ks_stat, pval = scipy.stats.kstest(sorted(samples), dist.cdf)
  plt.title("K-S stat: {}\np-value: {}".format(ks_stat, pval))
  savefig(model_dir, suffix)
  plt.close()


def savefig(model_dir, suffix):
  """Saves figure given suffix."""
  filename = os.path.join(model_dir, suffix)
  plt.savefig(filename)
  print("Figure saved in", filename)


class NutsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(NutsTest, self).setUp()
    self.model_dir = os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "test/")
    tf.io.gfile.makedirs(self.model_dir)

  def testOneStepFromOrigin(self):
    def target_log_prob_fn(event):
      return tfd.Normal(loc=0., scale=1.).log_prob(event)

    samples = []
    for seed in range(10):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[0.],
          step_size=[0.3],
          seed=seed)
      samples.append(state)

    samples = np.array(samples)
    plt.hist(samples, bins=30)
    savefig(self.model_dir, "one_step_from_origin.png")
    plt.close()

  def testReproducibility(self):
    def target_log_prob_fn(event):
      return tfd.Normal(loc=0., scale=1.).log_prob(event)

    tf.random.set_seed(4)
    xs = no_u_turn_sampler.kernel(
        target_log_prob_fn=target_log_prob_fn,
        current_state=[0.],
        step_size=[0.3],
        seed=3)
    tf.random.set_seed(4)
    ys = no_u_turn_sampler.kernel(
        target_log_prob_fn=target_log_prob_fn,
        current_state=[0.],
        step_size=[0.3],
        seed=3)
    for x, y in zip(xs, ys):
      self.assertAllEqual(x, y)

  def testNormal(self):
    def target_log_prob_fn(event):
      return tfd.Normal(loc=0., scale=1.).log_prob(event)

    rng = np.random.RandomState(seed=7)
    states = tf.cast(rng.normal(size=10), dtype=tf.float32)
    tf.random.set_seed(2)
    samples = []
    for seed, state in enumerate(states):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[state],
          step_size=[0.3],
          seed=seed)
      samples.append(state)

    samples = np.array(samples)
    plot_with_expectation(samples,
                          dist=scipy.stats.norm(0, 1),
                          model_dir=self.model_dir,
                          suffix="one_step_posterior_conservation_normal.png")

  def testLogitBeta(self):
    def target_log_prob_fn(event):
      return tfd.TransformedDistribution(
          distribution=tfd.Beta(concentration0=1.0, concentration1=3.0),
          bijector=tfb.Invert(tfb.Sigmoid())).log_prob(event)

    states = tfd.TransformedDistribution(
        distribution=tfd.Beta(concentration0=1.0, concentration1=3.0),
        bijector=tfb.Invert(tfb.Sigmoid())).sample(10, seed=7)
    plt.hist(states.numpy(), bins=30)
    savefig(self.model_dir, "logit_beta_start_positions.png")
    plt.close()

    samples = []
    for seed, state in enumerate(states):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[state],
          step_size=[0.3],
          seed=seed)
      samples.append(state)

    samples = np.array(samples)
    plt.hist(samples, bins=30)
    savefig(self.model_dir, "one_step_logit_beta_posterior_conservation.png")
    plt.close()

    _ = scipy.stats.ks_2samp(samples.flatten(), states.numpy().flatten())

  def testMultivariateNormal2d(self):
    def target_log_prob_fn(event):
      return tfd.MultivariateNormalFullCovariance(
          loc=tf.zeros(2), covariance_matrix=tf.eye(2)).log_prob(event)

    rng = np.random.RandomState(seed=7)
    states = tf.cast(rng.normal(size=[10, 2]), dtype=tf.float32)
    samples = []
    for seed, state in enumerate(states):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[state],
          step_size=[0.3],
          seed=seed)
      samples.append(state)

    samples = tf.stack(samples).numpy()
    plt.scatter(samples[:, 0], samples[:, 1])
    savefig(self.model_dir, "one_step_posterior_conservation_2d.png")
    plt.close()
    plot_with_expectation(samples[:, 0],
                          dist=scipy.stats.norm(0, 1),
                          model_dir=self.model_dir,
                          suffix="one_step_posterior_conservation_2d_dim_0.png")
    plot_with_expectation(samples[:, 1],
                          dist=scipy.stats.norm(0, 1),
                          model_dir=self.model_dir,
                          suffix="one_step_posterior_conservation_2d_dim_1.png")

  def testSkewedMultivariateNormal2d(self):
    def target_log_prob_fn(event):
      return tfd.MultivariateNormalFullCovariance(
          loc=tf.zeros(2),
          covariance_matrix=tf.linalg.diag([1., 10.])).log_prob(event)

    rng = np.random.RandomState(seed=7)
    states = tf.cast(rng.normal(scale=[1.0, 10.0], size=[10, 2]), tf.float32)
    plt.scatter(states[:, 0], states[:, 1])
    savefig(self.model_dir, "skewed_start_positions_2d.png")
    plt.close()

    samples = []
    for seed, state in enumerate(states):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[state],
          step_size=[0.3],
          seed=seed)
      samples.append(state)

    samples = tf.stack(samples).numpy()
    plt.scatter(samples[:, 0], samples[:, 1])
    savefig(self.model_dir, "one_step_skewed_posterior_conservation_2d.png")
    plt.close()
    plot_with_expectation(
        samples[:, 0],
        dist=scipy.stats.norm(0, 1),
        model_dir=self.model_dir,
        suffix="one_step_skewed_posterior_conservation_2d_dim_0.png")
    plot_with_expectation(
        samples[:, 1],
        dist=scipy.stats.norm(0, 10),
        model_dir=self.model_dir,
        suffix="one_step_skewed_posterior_conservation_2d_dim_1.png")

  @parameterized.parameters(
      (3, 10,),
      (5, 10,),
  )
  def testMultivariateNormalNd(self, event_size, num_samples):
    def target_log_prob_fn(event):
      return tfd.MultivariateNormalFullCovariance(
          loc=tf.zeros(event_size),
          covariance_matrix=tf.eye(event_size)).log_prob(event)

    state = tf.zeros(event_size)
    samples = []
    for seed in range(num_samples):
      [state], _, _ = no_u_turn_sampler.kernel(
          target_log_prob_fn=target_log_prob_fn,
          current_state=[state],
          step_size=[0.3],
          seed=seed)
      npstate = state.numpy()
      samples.append([npstate[0], npstate[1]])

    samples = np.array(samples)
    plt.scatter(samples[:, 0], samples[:, 1])
    savefig(self.model_dir,
            "projection_chain_{}d_normal_{}_steps.png".format(
                event_size, num_samples))
    plt.close()

    target_samples = tfd.MultivariateNormalFullCovariance(
        loc=tf.zeros(event_size),
        covariance_matrix=tf.eye(event_size)).sample(
            num_samples, seed=4).numpy()
    plt.scatter(target_samples[:, 0], target_samples[:, 1])
    savefig(self.model_dir,
            "projection_independent_{}d_normal_{}_samples.png".format(
                event_size, num_samples))
    plt.close()


if __name__ == "__main__":
  tf.test.main()
