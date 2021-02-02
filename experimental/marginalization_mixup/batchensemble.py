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

"""BatchEnsemble Wide ResNet 28-10 on CIFAR-10 and CIFAR-100."""

import functools
import os
import time

from absl import app
from absl import flags
from absl import logging
from experimental.marginalization_mixup import batchensemble_model  # local file import
from experimental.marginalization_mixup import data_utils  # local file import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.baselines.cifar import utils
import uncertainty_metrics as um

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 64,
                     'Batch size per TPU core/GPU. The number of new '
                     'datapoints gathered per batch is this number divided by '
                     'ensemble_size (we tile the batch by that # of times).')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('fast_weight_lr_multiplier', 1.0,
                   'fast weights lr multiplier.')
flags.DEFINE_float('train_proportion', default=1.0,
                   help='only use a proportion of training set.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['80', '160', '180'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 3e-4, 'L2 coefficient.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_float('diversity_coeff', 0., 'Diversity loss coefficient.')
flags.DEFINE_float('diversity_decay_epoch', 4, 'Diversity decay epoch.')
flags.DEFINE_integer('diversity_start_epoch', 100,
                     'Diversity loss starting epoch')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path', None,
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
flags.DEFINE_bool('random_augment', False,
                  'Whether random augment the input data.')
flags.DEFINE_bool('augmix', False, 'Whether augmix the input data.')
flags.DEFINE_integer('aug_count', 4,
                     'How many augmentations on input image;'
                     'It is ensemble_size if random_augment;'
                     'It is ensemble_size - 1 if augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth.')
flags.DEFINE_integer('augmix_width', 3, 'Augmix width.')
flags.DEFINE_float('mixup_alpha', 0.,
                   'Mixup hyperparameter, 0. to disable. Note 1. also disables '
                   'Mixup if adaptive_mixup is True.')
flags.DEFINE_bool('adaptive_mixup', False, 'Whether to use adaptive mixup.')
flags.DEFINE_bool('use_ensemble_bn', False, 'Whether to use ensemble bn.')
flags.DEFINE_float('label_smoothing', 0., 'Label smoothing.')
flags.DEFINE_bool('validation', False, 'Whether to use validation set.')
flags.DEFINE_bool('forget_mixup', False,
                  'Whether to mixup data based on forgetting counts. Only one '
                  'of the forget_mix or adaptive_mixup can be True.')
flags.DEFINE_integer('forget_threshold', 2, '1 / forget_threshold of training'
                     'examples will be applied mixup')
flags.DEFINE_bool('cutmix', False,
                  'Whether to use cutmix.')

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
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
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

  aug_params = {
      'augmix': FLAGS.augmix,
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth,
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width,
      'label_smoothing': FLAGS.label_smoothing,
      'ensemble_size': FLAGS.ensemble_size,
      'mixup_alpha': FLAGS.mixup_alpha,
      'random_augment': FLAGS.random_augment,
      'adaptive_mixup': FLAGS.adaptive_mixup,
      'forget_mixup': FLAGS.forget_mixup,
      'num_cores': FLAGS.num_cores,
      'threshold': FLAGS.forget_threshold,
      'cutmix': FLAGS.cutmix,
  }
  batch_size = ((FLAGS.per_core_batch_size // FLAGS.ensemble_size) *
                FLAGS.num_cores)
  train_input_fn = data_utils.load_input_fn(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      proportion=FLAGS.train_proportion,
      validation_set=FLAGS.validation,
      aug_params=aug_params)
  if FLAGS.validation:
    validation_input_fn = data_utils.load_input_fn(
        split=tfds.Split.VALIDATION,
        name=FLAGS.dataset,
        batch_size=FLAGS.per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        validation_set=True)
    val_dataset = strategy.experimental_distribute_datasets_from_function(
        validation_input_fn)
  clean_test_input_fn = data_utils.load_input_fn(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size // FLAGS.ensemble_size,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = strategy.experimental_distribute_dataset(
      train_input_fn())
  test_datasets = {
      'clean': strategy.experimental_distribute_datasets_from_function(
          clean_test_input_fn),
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
            batch_size=batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

  ds_info = tfds.builder(FLAGS.dataset).info
  num_train_examples = ds_info.splits['train'].num_examples
  # Train_proportion is a float so need to convert steps_per_epoch to int.
  if FLAGS.validation:
    # TODO(ywenxu): Remove hard-coding validation images.
    steps_per_epoch = int((num_train_examples *
                           FLAGS.train_proportion - 2500) // batch_size)
    steps_per_val = 2500 // (FLAGS.per_core_batch_size * FLAGS.num_cores)
  else:
    steps_per_epoch = int(
        num_train_examples * FLAGS.train_proportion) // batch_size
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  if FLAGS.use_bfloat16:
    policy = tf.python.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.python.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
    model = batchensemble_model.wide_resnet(
        input_shape=ds_info.features['image'].shape,
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        l2=FLAGS.l2,
        use_ensemble_bn=FLAGS.use_ensemble_bn)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.python.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)

    diversity_schedule = tf.python.keras.optimizers.schedules.ExponentialDecay(
        FLAGS.diversity_coeff, FLAGS.diversity_decay_epoch * steps_per_epoch,
        decay_rate=0.97, staircase=True)

    metrics = {
        'train/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'train/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.python.keras.metrics.Mean(),
        'train/similarity': tf.python.keras.metrics.Mean(),
        'train/l2': tf.python.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'test/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'test/member_accuracy_mean': (
            tf.python.keras.metrics.SparseCategoricalAccuracy()),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/member_ece_mean': um.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins)
    }
    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.python.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.python.keras.metrics.SparseCategoricalAccuracy())
      metrics['test/ece_member_{}'.format(i)] = (
          um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    test_diversity = {}
    training_diversity = {}
    corrupt_diversity = {}
    if FLAGS.ensemble_size > 1:
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
          corrupt_metrics['test/member_acc_mean_{}'.format(dataset_name)] = (
              tf.python.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/member_ece_mean_{}'.format(dataset_name)] = (
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
          corrupt_diversity['corrupt_diversity/average_kl_{}'.format(
              dataset_name)] = tf.python.keras.metrics.Mean()
          corrupt_diversity['corrupt_diversity/cosine_similarity_{}'.format(
              dataset_name)] = tf.python.keras.metrics.Mean()
          corrupt_diversity['corrupt_diversity/disagreement_{}'.format(
              dataset_name)] = tf.python.keras.metrics.Mean()

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
      if FLAGS.forget_mixup:
        images, labels, idx = inputs
      else:
        images, labels = inputs
      if FLAGS.adaptive_mixup or FLAGS.forget_mixup:
        images = tf.identity(images)
      elif FLAGS.augmix or FLAGS.random_augment:
        images_shape = tf.shape(images)
        images = tf.reshape(tf.transpose(
            images, [1, 0, 2, 3, 4]), [-1, images_shape[2],
                                       images_shape[3], images_shape[4]])
      else:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      # Augmix, adaptive mixup, forget mixup preprocessing gives tiled labels.
      if FLAGS.mixup_alpha > 0 or FLAGS.label_smoothing > 0 or FLAGS.cutmix:
        if FLAGS.augmix or FLAGS.adaptive_mixup or FLAGS.forget_mixup:
          labels = tf.identity(labels)
        else:
          labels = tf.tile(labels, [FLAGS.ensemble_size, 1])
      else:
        labels = tf.tile(labels, [FLAGS.ensemble_size])

      def _is_batch_norm(v):
        """Decide whether a variable belongs to `batch_norm`."""
        keywords = ['batchnorm', 'batch_norm', 'bn']
        return any([k in v.name.lower() for k in keywords])

      def _normalize(x):
        """Normalize an input with l2 norm."""
        l2 = tf.norm(x, ord=2, axis=-1)
        return x / tf.expand_dims(l2, axis=-1)

      # Taking the sum of upper triangular of XX^T and divided by ensemble size.
      def pairwise_cosine_distance(x):
        """Compute the pairwise distance in a matrix."""
        normalized_x = _normalize(x)
        return (tf.reduce_sum(
            tf.matmul(normalized_x, normalized_x, transpose_b=True)) -
                FLAGS.ensemble_size) / (2.0 * FLAGS.ensemble_size)

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        if FLAGS.mixup_alpha > 0 or FLAGS.label_smoothing > 0 or FLAGS.cutmix:
          negative_log_likelihood = tf.reduce_mean(
              tf.python.keras.losses.categorical_crossentropy(labels,
                                                       logits,
                                                       from_logits=True))
        else:
          negative_log_likelihood = tf.reduce_mean(
              tf.python.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))

        l2_loss = sum(model.losses)
        fast_weights = [var for var in model.trainable_variables if
                        not _is_batch_norm(var) and (
                            'alpha' in var.name or 'gamma' in var.name)]

        pairwise_distance_loss = tf.add_n(
            [pairwise_cosine_distance(var) for var in fast_weights])

        diversity_start_iter = steps_per_epoch * FLAGS.diversity_start_epoch
        diversity_iterations = optimizer.iterations - diversity_start_iter
        if diversity_iterations > 0:
          diversity_coeff = diversity_schedule(diversity_iterations)
          diversity_loss = diversity_coeff * pairwise_distance_loss
          loss = negative_log_likelihood + l2_loss + diversity_loss
        else:
          loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weight approximate
          # posterior/prior parameters. This is excludes BN and slow weights,
          # but pay caution to the naming scheme.
          if (not _is_batch_norm(var) and 'kernel' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      if FLAGS.ensemble_size > 1:
        per_probs = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        diversity_results = um.average_pairwise_diversity(
            per_probs, FLAGS.ensemble_size)
        for k, v in diversity_results.items():
          training_diversity['train/' + k].update_state(v)

      if FLAGS.mixup_alpha > 0 or FLAGS.label_smoothing > 0 or FLAGS.cutmix:
        labels = tf.argmax(labels, axis=-1)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/similarity'].update_state(pairwise_distance_loss)
      metrics['train/l2'].update_state(l2_loss)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)
      if FLAGS.forget_mixup:
        train_predictions = tf.argmax(probs, -1)
        labels = tf.cast(labels, train_predictions.dtype)
        # For each ensemble member, we accumulate the accuracy counts.
        accuracy_counts = tf.cast(tf.reshape(
            (train_predictions == labels), [FLAGS.ensemble_size, -1]),
                                  tf.float32)
        return accuracy_counts, idx

    if FLAGS.forget_mixup:
      return strategy.run(step_fn, args=(next(iterator),))
    else:
      strategy.run(step_fn, args=(next(iterator),))

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
      probs = tf.nn.softmax(logits)
      per_probs = tf.split(probs,
                           num_or_size_splits=FLAGS.ensemble_size,
                           axis=0)
      for i in range(FLAGS.ensemble_size):
        member_probs = per_probs[i]
        if dataset_name == 'clean':
          member_loss = tf.python.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)
          metrics['test/member_accuracy_mean'].update_state(
              labels, member_probs)
          metrics['test/ece_member_{}'.format(i)].update_state(labels,
                                                               member_probs)
          metrics['test/member_ece_mean'].update_state(labels, member_probs)
        elif dataset_name != 'validation':
          corrupt_metrics['test/member_acc_mean_{}'.format(
              dataset_name)].update_state(labels, member_probs)
          corrupt_metrics['test/member_ece_mean_{}'.format(
              dataset_name)].update_state(labels, member_probs)

      if FLAGS.ensemble_size > 1:
        per_probs_tensor = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        diversity_results = um.average_pairwise_diversity(
            per_probs_tensor, FLAGS.ensemble_size)
        if dataset_name == 'clean':
          for k, v in diversity_results.items():
            test_diversity['test/' + k].update_state(v)
        elif dataset_name != 'validation':
          for k, v in diversity_results.items():
            corrupt_diversity['corrupt_diversity/{}_{}'.format(
                k, dataset_name)].update_state(v)

      probs = tf.reduce_mean(per_probs, axis=0)
      negative_log_likelihood = tf.reduce_mean(
          tf.python.keras.losses.sparse_categorical_crossentropy(labels, probs))
      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
      elif dataset_name != 'validation':
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
            labels, probs)

      if dataset_name == 'validation':
        return per_probs_tensor, labels

    if dataset_name == 'validation':
      return strategy.run(step_fn, args=(next(iterator),))
    else:
      strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  start_time = time.time()
  forget_counts_history = []
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    acc_counts_list = []
    idx_list = []
    for step in range(steps_per_epoch):
      if FLAGS.forget_mixup:
        temp_accuracy_counts, temp_idx = train_step(train_iterator)
        acc_counts_list.append(temp_accuracy_counts)
        idx_list.append(temp_idx)
      else:
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
      if step % 20 == 0:
        logging.info(message)

    # Only one of the forget_mixup and adaptive_mixup can be true.
    if FLAGS.forget_mixup:
      current_acc = [tf.concat(list(acc_counts_list[i].values), axis=1)
                     for i in range(len(acc_counts_list))]
      total_idx = [tf.concat(list(idx_list[i].values), axis=0)
                   for i in range(len(idx_list))]
      current_acc = tf.cast(tf.concat(current_acc, axis=1), tf.int32)
      total_idx = tf.concat(total_idx, axis=0)

      current_forget_path = os.path.join(FLAGS.output_dir,
                                         'forget_counts.npy')
      last_acc_path = os.path.join(FLAGS.output_dir, 'last_acc.npy')
      if epoch == 0:
        forget_counts = tf.zeros(
            [FLAGS.ensemble_size, num_train_examples], dtype=tf.int32)
        last_acc = tf.zeros(
            [FLAGS.ensemble_size, num_train_examples], dtype=tf.int32)
      else:
        if 'last_acc' not in locals():
          with tf.io.gfile.GFile(last_acc_path, 'rb') as f:
            last_acc = np.load(f)
          last_acc = tf.cast(tf.convert_to_tensor(last_acc), tf.int32)
        if 'forget_counts' not in locals():
          with tf.io.gfile.GFile(current_forget_path, 'rb') as f:
            forget_counts = np.load(f)
          forget_counts = tf.cast(tf.convert_to_tensor(forget_counts), tf.int32)

      selected_last_acc = tf.gather(last_acc, total_idx, axis=1)
      forget_this_epoch = tf.cast(current_acc < selected_last_acc, tf.int32)
      forget_this_epoch = tf.transpose(forget_this_epoch)
      target_shape = tf.constant([num_train_examples, FLAGS.ensemble_size])
      current_forget_counts = tf.scatter_nd(tf.reshape(total_idx, [-1, 1]),
                                            forget_this_epoch, target_shape)
      current_forget_counts = tf.transpose(current_forget_counts)
      acc_this_epoch = tf.transpose(current_acc)
      last_acc = tf.scatter_nd(tf.reshape(total_idx, [-1, 1]),
                               acc_this_epoch, target_shape)
      # This is lower bound of true acc.
      last_acc = tf.transpose(last_acc)

      # TODO(ywenxu): We count the dropped examples as forget. Fix this later.
      forget_counts += current_forget_counts
      forget_counts_history.append(forget_counts)
      logging.info('forgetting counts')
      logging.info(tf.stack(forget_counts_history, 0))
      with tf.io.gfile.GFile(os.path.join(
          FLAGS.output_dir, 'forget_counts_history.npy'), 'wb') as f:
        np.save(f, tf.stack(forget_counts_history, 0).numpy())
      with tf.io.gfile.GFile(current_forget_path, 'wb') as f:
        np.save(f, forget_counts.numpy())
      with tf.io.gfile.GFile(last_acc_path, 'wb') as f:
        np.save(f, last_acc.numpy())
      aug_params['forget_counts_dir'] = current_forget_path

      train_input_fn = data_utils.load_input_fn(
          split=tfds.Split.TRAIN,
          name=FLAGS.dataset,
          batch_size=FLAGS.num_cores * (
              FLAGS.per_core_batch_size // FLAGS.ensemble_size),
          use_bfloat16=FLAGS.use_bfloat16,
          validation_set=FLAGS.validation,
          aug_params=aug_params)
      train_dataset = strategy.experimental_distribute_dataset(
          train_input_fn())
      train_iterator = iter(train_dataset)

    if FLAGS.adaptive_mixup:
      val_iterator = iter(val_dataset)
      logging.info('Testing on validation dataset')
      predictions_list = []
      labels_list = []
      for step in range(steps_per_val):
        temp_predictions, temp_labels = test_step(val_iterator, 'validation')
        predictions_list.append(temp_predictions)
        labels_list.append(temp_labels)
      predictions = [tf.concat(list(predictions_list[i].values), axis=1)
                     for i in range(len(predictions_list))]
      labels = [tf.concat(list(labels_list[i].values), axis=0)
                for i in range(len(labels_list))]
      predictions = tf.concat(predictions, axis=1)
      labels = tf.cast(tf.concat(labels, axis=0), tf.int64)

      def compute_acc_conf(preds, label, focus_class):
        class_preds = tf.boolean_mask(preds, label == focus_class, axis=1)
        class_pred_labels = tf.argmax(class_preds, axis=-1)
        confidence = tf.reduce_mean(tf.reduce_max(class_preds, axis=-1), -1)
        accuracy = tf.reduce_mean(tf.cast(
            class_pred_labels == focus_class, tf.float32), axis=-1)
        return accuracy - confidence

      calibration_per_class = [compute_acc_conf(
          predictions, labels, i) for i in range(num_classes)]
      calibration_per_class = tf.stack(calibration_per_class, axis=1)
      logging.info('calibration per class')
      logging.info(calibration_per_class)
      mixup_coeff = tf.where(calibration_per_class > 0, 1.0, FLAGS.mixup_alpha)
      mixup_coeff = tf.clip_by_value(mixup_coeff, 0, 1)
      logging.info('mixup coeff')
      logging.info(mixup_coeff)
      aug_params['mixup_coeff'] = mixup_coeff
      train_input_fn = data_utils.load_input_fn(
          split=tfds.Split.TRAIN,
          name=FLAGS.dataset,
          batch_size=FLAGS.per_core_batch_size // FLAGS.ensemble_size,
          use_bfloat16=FLAGS.use_bfloat16,
          validation_set=True,
          aug_params=aug_params)
      train_dataset = strategy.experimental_distribute_datasets_from_function(
          train_input_fn)
      train_iterator = iter(train_dataset)

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
      # This includes corrupt_diversity whose disagreement normalized by its
      # corrupt mean error rate.
      corrupt_results = utils.aggregate_corrupt_metrics(
          corrupt_metrics,
          corruption_types,
          max_intensity,
          corrupt_diversity=corrupt_diversity,
          output_dir=FLAGS.output_dir)

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

    total_metrics = metrics.copy()
    total_metrics.update(training_diversity)
    total_metrics.update(test_diversity)
    total_results = {name: metric.result()
                     for name, metric in total_metrics.items()}
    total_results.update(corrupt_results)
    # Normalize all disagreement metrics (training, testing) by test accuracy.
    # Disagreement on corrupt dataset is normalized by their own error rate.
    test_acc = total_metrics['test/accuracy'].result()
    for name, metric in total_metrics.items():
      if 'disagreement' in name:
        total_results[name] = metric.result() / test_acc

    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)
      if FLAGS.forget_mixup:
        tf.summary.histogram('forget_counts', forget_counts, step=epoch + 1)

    for metric in total_metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

    # Need to store the last but one checkpoint in adaptive mixup setup.
    if FLAGS.adaptive_mixup and epoch == (FLAGS.train_epochs - 2):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'last_but_one_checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  final_save_name = os.path.join(FLAGS.output_dir, 'model')
  model.save(final_save_name)
  logging.info('Saved model to %s', final_save_name)

if __name__ == '__main__':
  app.run(main)
