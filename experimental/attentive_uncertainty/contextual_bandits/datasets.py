# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

"""Generates and saves instances of wheel bandit problems.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from deep_contextual_bandits import synthetic_data_sampler  # local file import
gfile = tf.compat.v1.gfile

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    'num_instances',
    100,
    'Number of contexts.')
flags.DEFINE_integer(
    'num_contexts',
    80000,
    'Number of contexts.')
flags.DEFINE_string(
    'datadir',
    '/tmp/wheel_bandit/data/',
    'Directory for saving npz files.')
flags.DEFINE_list(
    'deltas',
    [0.5, 0.7, 0.9, 0.95, 0.99],
    'List of deltas for wheel bandit.')


def get_wheel_data(num_contexts, delta, seed):
  """Samples wheel bandit data according to the benchmark configuration."""
  mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
  std_v = [0.01, 0.01, 0.01, 0.01, 0.01]
  mu_large = 50
  std_large = 0.01

  # For reproducible generation.
  np.random.seed(int(100*delta) + seed)
  dataset, opt_wheel = synthetic_data_sampler.sample_wheel_bandit_data(
      num_contexts,
      delta,
      mean_v,
      std_v,
      mu_large,
      std_large)
  opt_rewards, opt_actions = opt_wheel
  return dataset, opt_rewards, opt_actions


def main(argv):
  del argv  # unused arg
  num_instances = FLAGS.num_instances
  num_contexts = FLAGS.num_contexts
  datadir = FLAGS.datadir
  deltas = FLAGS.deltas

  for delta in deltas:
    print('Delta', delta)
    for i in range(num_instances):
      print('Instance', i)
      dataset, opt_rewards, opt_actions = get_wheel_data(num_contexts, delta, i)
      filename = os.path.join(datadir, str(delta) + '_' + str(i) + '.npz')
      with gfile.GFile(filename, 'w') as f:
        np.savez(
            f,
            dataset=dataset,
            opt_rewards=opt_rewards,
            opt_actions=opt_actions)

if __name__ == '__main__':
  app.run(main)
