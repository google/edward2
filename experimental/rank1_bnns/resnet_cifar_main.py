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

# Lint as: python3
"""ResNet-32x4 with rank-1 distributions."""
import functools
import itertools
import os
import time
from absl import app
from absl import flags
from absl import logging

from experimental.rank1_bnns import resnet_cifar_model  # local file import
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.baselines.cifar import utils
import uncertainty_metrics as um

# ~24.4 steps per epoch for 4x4 TPU; per_core_batch_size=64; 300 epochs;
# so 2/3 of training time.
flags.DEFINE_integer('kl_annealing_steps', int(24.4 * 200),
                     'Number of steps over which to anneal the KL term to 1.')
flags.DEFINE_string('alpha_initializer', 'trainable_deterministic',
                    'Initializer name for the alpha parameters.')
flags.DEFINE_string('gamma_initializer', 'trainable_deterministic',
                    'Initializer name for the gamma parameters.')
flags.DEFINE_string('alpha_regularizer', None,
                    'Regularizer name for the alpha parameters.')
flags.DEFINE_string('gamma_regularizer', None,
                    'Regularizer name for the gamma parameters.')
flags.DEFINE_boolean('use_additive_perturbation', False,
                     'Use additive perturbations instead of multiplicative.')
flags.DEFINE_integer('num_train_samples', 1,
                     'Number of samples per example during training.')
flags.DEFINE_integer('num_eval_samples', 1,
                     'Number of samples per example during evaluation.')

# General model flags
flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_bool(
    'member_sampling', default=False,
    help=('Whether or not to sample a single ensemble member per step with '
          'which to compute the loss and derivatives.'))
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.5,
                   'fast weights lr multiplier.')
flags.DEFINE_bool('version2', True, 'Use ensemble version2.')
flags.DEFINE_bool(
    'expected_probs', default=False,
    help=('Whether or not to compute the loss over the per-example average of '
          'the predicted probabilities across the ensemble members.'))
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.1, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['80', '160', '180'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('dropout_rate', 0.,
                   'Dropout rate. Only used if alpha/gamma initializers are, '
                   'e.g., trainable normal with a fixed stddev.')
flags.DEFINE_float('l2', 2e-4, 'L2 coefficient.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path',
                    '',
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', 250,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused arg.

  tf.random.set_seed(FLAGS.seed)

  if FLAGS.version2:
    per_core_bs_train = FLAGS.per_core_batch_size // (FLAGS.ensemble_size *
                                                      FLAGS.num_train_samples)
    per_core_bs_eval = FLAGS.per_core_batch_size // (FLAGS.ensemble_size *
                                                     FLAGS.num_eval_samples)
  else:
    per_core_bs_train = FLAGS.per_core_batch_size // FLAGS.num_train_samples
    per_core_bs_eval = FLAGS.per_core_batch_size // FLAGS.num_eval_samples
  batch_size_train = per_core_bs_train * FLAGS.num_cores
  batch_size_eval = per_core_bs_eval * FLAGS.num_cores

  logging.info('Saving checkpoints at %s', FLAGS.output_dir)

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

  train_dataset = utils.load_dataset(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=batch_size_train,
      use_bfloat16=FLAGS.use_bfloat16,
      normalize=False)
  clean_test_dataset = utils.load_dataset(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=batch_size_eval,
      use_bfloat16=FLAGS.use_bfloat16,
      normalize=False)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  if FLAGS.corruptions_interval > 0:
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
            batch_size=batch_size_eval,
            use_bfloat16=FLAGS.use_bfloat16,
            normalize=False)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

  ds_info = tfds.builder(FLAGS.dataset).info
  train_dataset_size = ds_info.splits['train'].num_examples
  test_dataset_size = ds_info.splits['test'].num_examples
  num_classes = ds_info.features['label'].num_classes

  steps_per_epoch = train_dataset_size // batch_size_train
  steps_per_eval = test_dataset_size // batch_size_eval

  if FLAGS.use_bfloat16:
    policy = tf.python.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.python.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-32 model')
    model = resnet_cifar_model.rank1_resnet_v1(
        input_shape=ds_info.features['image'].shape,
        depth=32,
        num_classes=num_classes,
        width_multiplier=4,
        alpha_initializer=FLAGS.alpha_initializer,
        gamma_initializer=FLAGS.gamma_initializer,
        alpha_regularizer=FLAGS.alpha_regularizer,
        gamma_regularizer=FLAGS.gamma_regularizer,
        use_additive_perturbation=FLAGS.use_additive_perturbation,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        dropout_rate=FLAGS.dropout_rate)
    logging.info(model.summary())
    base_lr = FLAGS.base_learning_rate * batch_size_train / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.python.keras.optimizers.SGD(
        lr_schedule, momentum=0.9, nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'train/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.python.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'test/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/loss': tf.python.keras.metrics.Mean(),
    }
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.python.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.python.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    test_diversity = {}
    training_diversity = {}
    if FLAGS.ensemble_size > 1:
      for i in range(FLAGS.ensemble_size):
        metrics['test/nll_member_{}'.format(i)] = tf.python.keras.metrics.Mean()
        metrics['test/accuracy_member_{}'.format(i)] = (
            tf.python.keras.metrics.SparseCategoricalAccuracy())
      test_diversity = {
          'test/disagreement': tf.python.keras.metrics.Mean(),
          'test/average_kl': tf.python.keras.metrics.Mean(),
          'test/cosine_similarity': tf.python.keras.metrics.Mean(),
      }
      training_diversity = {
          'train/disagreement': tf.python.keras.metrics.Mean(),
          'train/average_kl': tf.python.keras.metrics.Mean(),
          'train/cosine_similarity': tf.python.keras.metrics.Mean(),
      }

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      if FLAGS.version2 and FLAGS.ensemble_size > 1:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
        if not (FLAGS.member_sampling or FLAGS.expected_probs):
          labels = tf.tile(labels, [FLAGS.ensemble_size])

      if FLAGS.num_train_samples > 1:
        images = tf.tile(images, [FLAGS.num_train_samples, 1, 1, 1])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        probs = tf.nn.softmax(logits)
        # Diversity evaluation.
        if FLAGS.version2 and FLAGS.ensemble_size > 1:
          per_probs = tf.reshape(
              probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))

          diversity_results = um.average_pairwise_diversity(
              per_probs, FLAGS.ensemble_size)

        if FLAGS.num_train_samples > 1:
          probs = tf.reshape(probs,
                             tf.concat([[FLAGS.num_train_samples, -1],
                                        probs.shape[1:]], 0))
          probs = tf.reduce_mean(probs, 0)

        if FLAGS.member_sampling and FLAGS.version2 and FLAGS.ensemble_size > 1:
          idx = tf.random.uniform([], maxval=FLAGS.ensemble_size,
                                  dtype=tf.int64)
          idx_one_hot = tf.expand_dims(tf.one_hot(idx, FLAGS.ensemble_size,
                                                  dtype=probs.dtype), 0)
          probs_shape = probs.shape
          probs = tf.reshape(probs, [FLAGS.ensemble_size, -1])
          probs = tf.matmul(idx_one_hot, probs)
          probs = tf.reshape(probs, tf.concat([[-1], probs_shape[1:]], 0))

        elif FLAGS.expected_probs and FLAGS.version2 and FLAGS.ensemble_size > 1:
          probs = tf.reshape(probs,
                             tf.concat([[FLAGS.ensemble_size, -1],
                                        probs.shape[1:]], 0))
          probs = tf.reduce_mean(probs, 0)

        negative_log_likelihood = tf.reduce_mean(
            tf.python.keras.losses.sparse_categorical_crossentropy(labels, probs))

        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the slow weights and bias terms. This excludes BN
          # parameters and fast weight approximate posterior/prior parameters,
          # but pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        kl = sum(model.losses) / train_dataset_size
        kl_scale = tf.cast(optimizer.iterations + 1, kl.dtype)
        kl_scale /= FLAGS.kl_annealing_steps
        kl_scale = tf.minimum(1., kl_scale)
        kl_loss = kl_scale * kl

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss + kl_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      grad_list = []
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = list(zip(grads, model.trainable_variables))
        for vec, var in grads_and_vars:
          # Apply different learning rate on the fast weight approximate
          # posterior/prior parameters. This is excludes BN and slow weights,
          # but pay caution to the naming scheme.
          if ('batch_norm' not in var.name and 'kernel' not in var.name):
            grad_list.append((vec * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grad_list.append((vec, var))
        optimizer.apply_gradients(grad_list)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      if FLAGS.version2 and FLAGS.ensemble_size > 1:
        for k, v in diversity_results.items():
          training_diversity['train/' + k].update_state(v)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      if FLAGS.ensemble_size > 1:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      if FLAGS.num_eval_samples > 1:
        images = tf.tile(images, [FLAGS.num_eval_samples, 1, 1, 1])
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)

      if FLAGS.num_eval_samples > 1:
        probs = tf.reshape(probs,
                           tf.concat([[FLAGS.num_eval_samples, -1],
                                      probs.shape[1:]], 0))
        probs = tf.reduce_mean(probs, 0)

      if FLAGS.ensemble_size > 1:
        per_probs = tf.split(probs,
                             num_or_size_splits=FLAGS.ensemble_size,
                             axis=0)
        if dataset_name == 'clean':
          per_probs_tensor = tf.reshape(
              probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
          diversity_results = um.average_pairwise_diversity(
              per_probs_tensor, FLAGS.ensemble_size)

          for k, v in diversity_results.items():
            test_diversity['test/' + k].update_state(v)

          for i in range(FLAGS.ensemble_size):
            member_probs = per_probs[i]
            member_nll = tf.python.keras.losses.sparse_categorical_crossentropy(
                labels, member_probs)
            metrics['test/nll_member_{}'.format(i)].update_state(member_nll)
            metrics['test/accuracy_member_{}'.format(i)].update_state(
                labels, member_probs)

        probs = tf.reduce_mean(per_probs, axis=0)

      negative_log_likelihood = tf.reduce_mean(
          tf.python.keras.losses.sparse_categorical_crossentropy(labels, probs))
      filtered_variables = []
      for var in model.trainable_variables:
        if 'kernel' in var.name or 'bias' in var.name:
          filtered_variables.append(tf.reshape(var, (-1,)))

      kl = sum(model.losses) / test_dataset_size
      l2_loss = kl + FLAGS.l2 * 2 * tf.nn.l2_loss(
          tf.concat(filtered_variables, axis=0))
      loss = negative_log_likelihood + l2_loss
      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
        metrics['test/loss'].update_state(loss)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
            labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      train_step(train_iterator)

      current_step = epoch * steps_per_epoch + (step + 1)
      max_steps = steps_per_epoch * FLAGS.train_epochs
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                     current_step / max_steps,
                     epoch + 1,
                     FLAGS.train_epochs,
                     steps_per_sec,
                     eta_seconds / 60,
                     time_elapsed / 60))
      work_unit.set_notes(message)
      if step % 20 == 0:
        logging.info(message)

    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        test_step(test_iterator, dataset_name)
      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types,
                                                        max_intensity)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    for i in range(FLAGS.ensemble_size):
      logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%',
                   i, metrics['test/nll_member_{}'.format(i)].result(),
                   metrics['test/accuracy_member_{}'.format(i)].result() * 100)
    total_metrics = itertools.chain(metrics.items(),
                                    training_diversity.items(),
                                    test_diversity.items())
    total_results = {name: metric.result() for name, metric in total_metrics}
    total_results.update(corrupt_results)
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for name, result in total_results.items():
      name = name.replace('/', '_')
      if 'negative_log_likelihood' in name:
        # Plots sort WIDs from high-to-low so look at maximization objectives.
        name = name.replace('negative_log_likelihood', 'log_likelihood')
        result = -result
      objective = work_unit.get_measurement_series(name)
      objective.create_measurement(result, epoch + 1)

    for _, metric in total_metrics:
      metric.reset_states()
    summary_writer.flush()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  app.run(main)
