# coding=utf-8
# Copyright 2023 The Edward2 Authors.
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

# pytype: disable=attribute-error
"""Benchmark script for the wheel bandit task.
"""

import os
import pickle
import time

from absl import app
from absl import flags
from experimental.attentive_uncertainty import attention  # local file import
from experimental.attentive_uncertainty.contextual_bandits import offline_contextual_bandits  # local file import
from experimental.attentive_uncertainty.contextual_bandits import online_contextual_bandits  # local file import
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
    '/tmp/wheel_bandit/results/',
    'Directory with saved pkl files for full results.')
flags.DEFINE_string(
    'ckptdir',
    '/tmp/wheel_bandit/ckpts/',
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
    'best_',
    'Prefix of best model ckpts.')
flags.DEFINE_string(
    'suffix',
    '_mse.ckpt',
    'Suffix of best model ckpts.')
flags.DEFINE_list(
    'algo_names',
    ['uniform', 'snp_posterior_gp_offline'],
    'List of algorithms to benchmark.')

context_dim = 2
num_actions = 5


def run_contextual_bandit(dataset, algos, save_once=False, pkl_file=None):
  """Run a contextual bandit problem on a set of algorithms.

  Args:
    dataset: Matrix where every row is a context + num_actions rewards.
    algos: List of algorithms to use in the contextual bandit instance.
    save_once: True if state has been saved once before
    pkl_file: pickle file for saving state.

  Returns:
    h_actions: Matrix with actions: size (num_context, num_algorithms).
    h_rewards: Matrix with rewards: size (num_context, num_algorithms).
  """

  num_contexts = dataset.shape[0]

  # Create contextual bandit
  cmab = contextual_bandit.ContextualBandit(context_dim, num_actions)
  cmab.feed_data(dataset)
  if not save_once or pkl_file is None:
    h_actions = np.empty((0, len(algos)), float)
    h_rewards = np.empty((0, len(algos)), float)

    start_context = 0
  else:
    with gfile.Open(pkl_file, 'rb') as infile:
      saved_state = pickle.load(infile)
      start_context = saved_state['start_context']
      algos[0].data_h.replace_data(
          saved_state['contexts'],
          saved_state['actions'],
          saved_state['rewards'])
      h_actions = saved_state['h_actions']
      h_rewards = saved_state['h_rewards']

  # Run the contextual bandit process
  for i in range(start_context, num_contexts):
    context = cmab.context(i)
    actions = [a.action(context) for a in algos]
    rewards = [cmab.reward(i, action) for action in actions]

    for j, a in enumerate(algos):
      a.update(context, actions[j], rewards[j])

    h_actions = np.vstack((h_actions, np.array(actions)))
    h_rewards = np.vstack((h_rewards, np.array(rewards)))

    if (i+1) % 500 == 0 and pkl_file is not None:
      savedict = {'h_rewards': h_rewards,
                  'h_actions': h_actions,
                  'contexts': algos[0].data_h.contexts,
                  'actions': algos[0].data_h.actions,
                  'rewards': algos[0].data_h.rewards,
                  'start_context': i+1}
      with gfile.Open(pkl_file, 'wb') as outfile:
        pickle.dump(savedict, outfile)

  return h_actions, h_rewards


def run_trial(trial_idx, delta, algo_names):
  """Runs a trial of wheel bandit problem instance for a set of algorithms."""

  all_algo_names = '_'.join(algo_names)
  runfile = str(delta) + '_' + str(trial_idx) + '_' + all_algo_names + '.pkl'
  savefile = os.path.join(FLAGS.savedir, runfile)
  if gfile.Exists(savefile):
    print('File exists...terminating')
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
    elif algo_name[:3] == 'snp' or algo_name[:3] == 'anp':
      hidden_size = 64
      latent_units = 32
      global_latent_net_sizes = [hidden_size]*2 + [2*latent_units]
      if algo_name[:3] == 'snp':
        local_latent_net_sizes = [hidden_size]*3 + [2]
      else:
        local_latent_net_sizes = [hidden_size]*3 + [2*5]
      x_y_encoder_sizes = [hidden_size]*3
      heteroskedastic_net_sizes = None
      mean_att_type = attention.laplace_attention
      scale_att_type_1 = attention.laplace_attention
      scale_att_type_2 = attention.laplace_attention
      att_type = 'multihead'
      att_heads = 8
      data_uncertainty = False
      is_anp = True if algo_name[:3] == 'anp' else False

      hparams = contrib_training.HParams(
          num_actions=num_actions,
          context_dim=context_dim,
          init_scale=0.3,
          activation=tf.nn.relu,
          output_activation=tf.nn.relu,
          x_encoder_sizes=x_encoder_sizes,
          x_y_encoder_sizes=x_y_encoder_sizes,
          global_latent_net_sizes=global_latent_net_sizes,
          local_latent_net_sizes=local_latent_net_sizes,
          heteroskedastic_net_sizes=heteroskedastic_net_sizes,
          att_type=att_type,
          att_heads=att_heads,
          mean_att_type=mean_att_type,
          scale_att_type_1=scale_att_type_1,
          scale_att_type_2=scale_att_type_2,
          data_uncertainty=data_uncertainty,
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
          training_epochs=50,
          uncertainty_type='attentive_freeform',
          local_variational=True,
          model_path=None,
          is_anp=is_anp)

      config = algo_name.split('_')
      if config[1] == 'prior':
        hparams.set_hparam('local_variational', False)

      if config[2] == 'gp':
        hparams.set_hparam('uncertainty_type', 'attentive_gp')

      if config[3] == 'warmstart' or config[3] == 'offline':
        mfile = FLAGS.prefix + config[1] + '_' + config[2] + FLAGS.suffix
        if algo_name[:3] == 'anp':
          mfile = 'anp_' + mfile
        mpath = os.path.join(FLAGS.modeldir, mfile)
        hparams.set_hparam('model_path', mpath)

      if config[3] == 'online' or config[3] == 'warmstart':
        algos.append(online_contextual_bandits.OnlineContextualBandits(
            algo_name, hparams))
      else:
        algos.append(offline_contextual_bandits.OfflineContextualBandits(
            algo_name, hparams))
        ckptfile = os.path.join(FLAGS.ckptdir, runfile)
        if gfile.Exists(ckptfile):
          save_once = True

  t_init = time.time()
  print('started')
  _, h_rewards = run_contextual_bandit(
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
