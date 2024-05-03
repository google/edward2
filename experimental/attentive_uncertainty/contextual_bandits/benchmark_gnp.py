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

"""Benchmark script for the wheel bandit task.
"""

import os
import pickle
import time

from absl import app
from absl import flags
from experimental.attentive_uncertainty.contextual_bandits import offline_contextual_bandits_gnp  # local file import
import numpy as np
import tensorflow.compat.v1 as tf

from deep_contextual_bandits import contextual_bandit  # local file import
from deep_contextual_bandits import neural_linear_sampling  # local file import
from deep_contextual_bandits import posterior_bnn_sampling  # local file import
from deep_contextual_bandits import uniform_sampling  # local file import

from tensorflow.contrib import training as contrib_training

gfile = tf.compat.v1.gfile

tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string(
    'logdir',
    '/tmp/bandits/',
    'Base directory to save output.')
flags.DEFINE_integer(
    'trial_idx',
    0,
    'Rerun idx of problem instance.')
flags.DEFINE_float(
    'delta',
    0.5,
    'delta parameter for wheel bandit instance.')
flags.DEFINE_string(
    'modeldir',
    '/tmp/wheel_bandit/models/multitask',
    'Directory with pretrained models.')
flags.DEFINE_string(
    'savedir',
    '/tmp/wheel_bandit/results/gnp',
    'Directory with saved pkl files for full results.')
flags.DEFINE_string(
    'ckptdir',
    '/tmp/wheel_bandit/ckpts/gnp',
    'Directory with saved pkl files for full ckpts.')
flags.DEFINE_string(
    'datasetdir',
    '/tmp/wheel_bandit/data/',
    'Directory with saved data instances.')
flags.DEFINE_integer(
    'exp_idx',
    0,
    'Experiment idx of full run.')
flags.DEFINE_string(
    'prefix',
    'gnp_',
    'Prefix of best model ckpts.')
flags.DEFINE_string(
    'suffix',
    '.ckpt',
    'Suffix of best model ckpts.')
flags.DEFINE_list(
    'algo_names',
    ['uniform', 'gnp_acns'],
    'List of algorithms to benchmark.')

context_dim = 2
num_actions = 5


def run_trial(trial_idx, delta, algo_names):
  """Runs a trial of wheel bandit problem instance for a set of algorithms."""

  all_algo_names = '_'.join(algo_names)
  runfile = str(delta) + '_' + str(trial_idx) + '_' + all_algo_names + '.pkl'
  savefile = os.path.join(FLAGS.savedir, runfile)
  if gfile.Exists(savefile):
    print('File exists...terminating')
    print(savefile)
    with gfile.Open(savefile, 'rb') as infile:
      saved_state = pickle.load(infile, encoding='latin-1')
    return saved_state['h_rewards'], saved_state['time']

  filename = os.path.join(
      FLAGS.datasetdir,
      str(delta) + '_' + str(trial_idx) + '.npz')
  with gfile.GFile(filename, 'r') as f:
    sampled_vals = np.load(f)
    dataset = sampled_vals['dataset']

  x_hidden_size = 100
  x_encoder_sizes = [x_hidden_size]*2

  algos = []
  ckptfile = None
  save_once = False
  for algo_name in algo_names:
    if algo_name == 'uniform':
      hparams = contrib_training.HParams(num_actions=num_actions)
      algos.append(uniform_sampling.UniformSampling(algo_name, hparams))
    elif algo_name == 'neurolinear':
      hparams = contrib_training.HParams(
          num_actions=num_actions,
          context_dim=context_dim,
          init_scale=0.3,
          activation=tf.nn.relu,
          output_activation=tf.nn.relu,
          layer_sizes=x_encoder_sizes,
          batch_size=512,
          activate_decay=True,
          initial_lr=0.1,
          max_grad_norm=5.0,
          show_training=False,
          freq_summary=1000,
          buffer_s=-1,
          initial_pulls=2,
          reset_lr=True,
          lr_decay_rate=0.5,
          training_freq=1,
          training_freq_network=20,
          training_epochs=50,
          a0=12,
          b0=30,
          lambda_prior=23)
      algos.append(neural_linear_sampling.NeuralLinearPosteriorSampling(
          algo_name, hparams))
    elif algo_name == 'multitaskgp':
      hparams_gp = contrib_training.HParams(
          num_actions=num_actions,
          num_outputs=num_actions,
          context_dim=context_dim,
          reset_lr=False,
          learn_embeddings=True,
          max_num_points=1000,
          show_training=False,
          freq_summary=1000,
          batch_size=512,
          keep_fixed_after_max_obs=True,
          training_freq=20,
          initial_pulls=2,
          training_epochs=50,
          lr=0.01,
          buffer_s=-1,
          initial_lr=0.001,
          lr_decay_rate=0.0,
          optimizer='RMS',
          task_latent_dim=5,
          activate_decay=False)
      algos.append(posterior_bnn_sampling.PosteriorBNNSampling(
          algo_name, hparams_gp, 'GP'))
    elif algo_name[:3] == 'gnp':
      hidden_size = 64
      x_encoder_net_sizes = None
      decoder_net_sizes = [hidden_size]*3 + [2*num_actions]
      heteroskedastic_net_sizes = None
      att_type = 'multihead'
      att_heads = 8
      data_uncertainty = False

      config = algo_name.split('_')
      model_type = config[1]

      if algo_name[:len('gnp_anp_beta_')] == 'gnp_anp_beta_':
        mfile = algo_name + FLAGS.suffix
        x_y_encoder_net_sizes = [hidden_size]*3
        global_latent_net_sizes = [hidden_size]*2
        local_latent_net_sizes = None
        beta = float(config[3])
        temperature = float(config[5])
      else:
        mfile = FLAGS.prefix + config[1] + FLAGS.suffix

        if model_type == 'cnp':
          x_y_encoder_net_sizes = [hidden_size]*4
          global_latent_net_sizes = None
          local_latent_net_sizes = None
        elif model_type == 'np':
          x_y_encoder_net_sizes = [hidden_size]*2
          global_latent_net_sizes = [hidden_size]*2
          local_latent_net_sizes = None
        elif model_type == 'anp':
          x_y_encoder_net_sizes = [hidden_size]*2
          global_latent_net_sizes = [hidden_size]*2
          local_latent_net_sizes = None
        elif model_type == 'acnp':
          x_y_encoder_net_sizes = [hidden_size]*4
          global_latent_net_sizes = None
          local_latent_net_sizes = None
        elif model_type == 'acns':
          x_y_encoder_net_sizes = [hidden_size]*2
          global_latent_net_sizes = [hidden_size]*2
          local_latent_net_sizes = [hidden_size]*2

        beta = 1.
        temperature = 1.

      mpath = os.path.join(FLAGS.modeldir, mfile)

      hparams = contrib_training.HParams(
          num_actions=num_actions,
          context_dim=context_dim,
          init_scale=0.3,
          activation=tf.nn.relu,
          output_activation=tf.nn.relu,
          x_encoder_net_sizes=x_encoder_net_sizes,
          x_y_encoder_net_sizes=x_y_encoder_net_sizes,
          global_latent_net_sizes=global_latent_net_sizes,
          local_latent_net_sizes=local_latent_net_sizes,
          decoder_net_sizes=decoder_net_sizes,
          heteroskedastic_net_sizes=heteroskedastic_net_sizes,
          att_type=att_type,
          att_heads=att_heads,
          model_type=model_type,
          data_uncertainty=data_uncertainty,
          beta=beta,
          temperature=temperature,
          model_path=mpath,
          batch_size=512,
          activate_decay=True,
          initial_lr=0.1,
          max_grad_norm=5.0,
          show_training=False,
          freq_summary=1000,
          buffer_s=-1,
          initial_pulls=2,
          reset_lr=True,
          lr_decay_rate=0.5,
          training_freq=10,
          training_freq_network=20,
          training_epochs=50)
      algos.append(offline_contextual_bandits_gnp.OfflineContextualBandits(
          algo_name, hparams))
      ckptfile = os.path.join(FLAGS.ckptdir, runfile)
      if gfile.Exists(ckptfile):
        save_once = True

  t_init = time.time()
  print('started')
  print([algo.name for algo in algos])
  _, h_rewards = contextual_bandit.run_contextual_bandit_new(  # pytype: disable=module-attr
      context_dim,
      num_actions,
      dataset,
      algos,
      save_once=save_once,
      pkl_file=ckptfile)
  t_final = time.time()

  savedict = {'h_rewards': h_rewards, 'time': t_final-t_init}
  with gfile.Open(savefile, 'wb') as outfile:
    pickle.dump(savedict, outfile)
  return h_rewards, t_final - t_init


def main(_):
  run_trial(FLAGS.trial_idx, FLAGS.delta, FLAGS.algo_names)

if __name__ == '__main__':
  app.run(main)
