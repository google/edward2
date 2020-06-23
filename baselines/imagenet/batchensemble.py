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

"""Batch Ensemble ResNet-50."""

import os
import time

from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import batchensemble_model  # local file import
import utils  # local file import
from edward2.google.rank1_pert.ensemble_keras import utils as be_utils
import tensorflow.compat.v2 as tf

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.5,
                   'fast weights lr multiplier.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string(
    'output_dir', '/tmp/imagenet', 'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 135, 'Number of training epochs.')
flags.DEFINE_integer(
    'corruptions_interval', 135,
    'Number of epochs between evaluating on the corrupted '
    'test data. Use -1 to never evaluate.')
flags.DEFINE_integer(
    'checkpoint_interval', 27,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')
flags.DEFINE_bool('use_ensemble_bn', False, 'Whether to use ensemble bn.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_string('similarity_metric', 'cosine', 'Similarity metric in '
                    '[cosine, dpp_logdet]')
flags.DEFINE_string('dpp_kernel', 'linear', 'Kernel for DPP log determinant')
flags.DEFINE_bool('use_output_similarity', False,
                  'If true, compute similarity on the ensemble outputs.')
flags.DEFINE_enum('diversity_scheduler', 'LinearAnnealing',
                  ['LinearAnnealing', 'ExponentialDecay', 'Fixed'],
                  'Diversity coefficient scheduler..')
flags.DEFINE_float('annealing_epochs', 200,
                   'Number of epochs over which to linearly anneal')
flags.DEFINE_float('diversity_coeff', 0., 'Diversity loss coefficient.')
flags.DEFINE_float('diversity_decay_epoch', 4, 'Diversity decay epoch.')
flags.DEFINE_float('diversity_decay_rate', 0.97, 'Rate of exponential decay.')
flags.DEFINE_integer('diversity_start_epoch', 100,
                     'Diversity loss starting epoch')

FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000

_LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  per_core_batch_size = FLAGS.per_core_batch_size // FLAGS.ensemble_size
  batch_size = per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  imagenet_train = utils.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  imagenet_eval = utils.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  test_datasets = {
      'clean':
          strategy.experimental_distribute_datasets_from_function(
              imagenet_eval.input_fn),
  }
  if FLAGS.corruptions_interval > 0:
    corruption_types, max_intensity = utils.load_corrupted_test_info()
    for name in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset_name = '{0}_{1}'.format(name, intensity)
        corrupt_input_fn = utils.corrupt_test_input_fn(
            batch_size=per_core_batch_size,
            corruption_name=name,
            corruption_intensity=intensity,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets[dataset_name] = (
            strategy.experimental_distribute_datasets_from_function(
                corrupt_input_fn))

  train_dataset = strategy.experimental_distribute_datasets_from_function(
      imagenet_train.input_fn)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = batchensemble_model.ensemble_resnet50(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        use_ensemble_bn=FLAGS.use_ensemble_bn)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 256
    learning_rate = utils.LearningRateSchedule(steps_per_epoch, base_lr,
                                               FLAGS.train_epochs, _LR_SCHEDULE)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.9, nesterov=True)

    if FLAGS.diversity_scheduler == 'ExponentialDecay':
      diversity_schedule = be_utils.ExponentialDecay(
          initial_coeff=FLAGS.diversity_coeff,
          start_epoch=FLAGS.diversity_start_epoch,
          decay_epoch=FLAGS.diversity_decay_epoch,
          steps_per_epoch=steps_per_epoch,
          decay_rate=FLAGS.diversity_decay_rate,
          staircase=True)

    elif FLAGS.diversity_scheduler == 'LinearAnnealing':
      diversity_schedule = be_utils.LinearAnnealing(
          initial_coeff=FLAGS.diversity_coeff,
          annealing_epochs=FLAGS.annealing_epochs,
          steps_per_epoch=steps_per_epoch)
    else:
      diversity_schedule = lambda x: FLAGS.diversity_coeff

    metrics = {
        'train/similarity_loss':
            tf.keras.metrics.Mean(),
        'train/weights_similarity':
            tf.keras.metrics.Mean(),
        'train/outputs_similarity':
            tf.keras.metrics.Mean(),
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss':
            tf.keras.metrics.Mean(),
        'train/ece':
            ed.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece':
            ed.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/weights_similarity':
            tf.keras.metrics.Mean(),
        'test/outputs_similarity':
            tf.keras.metrics.Mean()
    }
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              ed.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    test_diversity = {}
    training_diversity = {}
    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())
    test_diversity = {
        'test/disagreement': tf.keras.metrics.Mean(),
        'test/average_kl': tf.keras.metrics.Mean(),
        'test/cosine_similarity': tf.keras.metrics.Mean(),
    }
    training_diversity = {
        'train/disagreement': tf.keras.metrics.Mean(),
        'train/average_kl': tf.keras.metrics.Mean(),
        'train/cosine_similarity': tf.keras.metrics.Mean(),
    }

    logging.info('Finished building Keras ResNet-50 model')

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
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      labels = tf.tile(labels, [FLAGS.ensemble_size])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        probs = tf.nn.softmax(logits)
        per_probs = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        diversity_results = ed.metrics.average_pairwise_diversity(
            per_probs, FLAGS.ensemble_size)

        # print(' > per_probs {}'.format(per_probs))
        similarity_coeff, similarity_loss = be_utils.scaled_similarity_loss(
            FLAGS.diversity_coeff, diversity_schedule, optimizer.iterations,
            FLAGS.similarity_metric, FLAGS.dpp_kernel,
            model.trainable_variables, FLAGS.use_output_similarity, per_probs)
        weights_similarity = be_utils.fast_weights_similarity(
            model.trainable_variables, FLAGS.similarity_metric,
            FLAGS.dpp_kernel)
        outputs_similarity = be_utils.outputs_similarity(
            per_probs, FLAGS.similarity_metric, FLAGS.dpp_kernel)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True))
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the slow weights and bias terms. This excludes BN
          # parameters and fast weight approximate posterior/prior parameters,
          # but pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        loss = negative_log_likelihood + l2_loss + similarity_coeff * similarity_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weights. This excludes BN
          # and slow weights, but pay caution to the naming scheme.
          if ('batch_norm' not in var.name and 'kernel' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      metrics['train/similarity_loss'].update_state(similarity_coeff *
                                                    similarity_loss)
      metrics['train/weights_similarity'].update_state(weights_similarity)
      metrics['train/outputs_similarity'].update_state(outputs_similarity)

      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)
      for k, v in diversity_results.items():
        training_diversity['train/' + k].update_state(v)

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

      if dataset_name == 'clean':
        per_probs_tensor = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        diversity_results = ed.metrics.average_pairwise_diversity(
            per_probs_tensor, FLAGS.ensemble_size)
        outputs_similarity = be_utils.outputs_similarity(
            per_probs_tensor, FLAGS.similarity_metric, FLAGS.dpp_kernel)
        for k, v in diversity_results.items():
          test_diversity['test/' + k].update_state(v)

      per_probs = tf.split(
          probs, num_or_size_splits=FLAGS.ensemble_size, axis=0)
      probs = tf.reduce_mean(per_probs, axis=0)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        for i in range(FLAGS.ensemble_size):
          member_probs = per_probs[i]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)

        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
        weights_similarity = be_utils.fast_weights_similarity(
            model.trainable_variables, FLAGS.similarity_metric,
            FLAGS.dpp_kernel)
        metrics['test/weights_similarity'].update_state(weights_similarity)
        metrics['test/outputs_similarity'].update_state(outputs_similarity)
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
                     current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                     steps_per_sec, eta_seconds / 60, time_elapsed / 60))
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
          logging.info('Starting to run eval step %s of epoch: %s', step, epoch)
        test_step(test_iterator, dataset_name)
      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(
          corrupt_metrics, corruption_types, max_intensity,
          FLAGS.alexnet_errors_path)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    for i in range(FLAGS.ensemble_size):
      logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%', i,
                   metrics['test/nll_member_{}'.format(i)].result(),
                   metrics['test/accuracy_member_{}'.format(i)].result() * 100)

    total_metrics = metrics.copy()
    total_metrics.update(training_diversity)
    total_metrics.update(test_diversity)
    total_results = {
        name: metric.result() for name, metric in total_metrics.items()
    }
    total_results.update(corrupt_results)
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for _, metric in total_metrics.items():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)


if __name__ == '__main__':
  app.run(main)
