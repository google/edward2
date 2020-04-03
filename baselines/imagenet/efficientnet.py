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

"""ResNet-50 with rank-1 priors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import efficientnet_builder  # local file import
import utils  # local file import
import tensorflow.compat.v2 as tf

# ~312.78 steps per epoch for 4x4 TPU; per_core_batch_size=128; 350 epochs;

# General model flags
flags.DEFINE_enum('model_name',
                  default='efficientnet-b0',
                  enum_values=['efficientnet-b0', 'efficientnet-b1',
                               'efficientnet-b2', 'efficientnet-b3'],
                  help='Efficientnet model name.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.016,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('l2', 5e-6, 'L2 coefficient.')
flags.DEFINE_string('data_dir', '', 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 350, 'Number of training epochs.')
flags.DEFINE_integer('checkpoint_interval', 15,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 5, 'How many epochs to run test.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing constant.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
IMAGENET_VALIDATION_IMAGES = 50000


def main(argv):
  del argv  # unused arg
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)

  per_core_batch_size = FLAGS.per_core_batch_size

  batch_size = per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size
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
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  _, _, input_image_size, _ = efficientnet_builder.efficientnet_params(
      FLAGS.model_name)
  imagenet_train = utils.ImageNetInput(
      is_training=True,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=FLAGS.data_dir,
      batch_size=FLAGS.per_core_batch_size,
      image_size=input_image_size,
      normalize_input=True,
      one_hot=True)
  imagenet_eval = utils.ImageNetInput(
      is_training=False,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      image_size=input_image_size,
      normalize_input=True,
      one_hot=True)
  train_dataset = strategy.experimental_distribute_datasets_from_function(
      imagenet_train.input_fn)
  test_datasets = {
      'clean':
          strategy.experimental_distribute_dataset(imagenet_eval.input_fn()),
  }
  train_iterator = iter(train_dataset)
  test_iterator = iter(test_datasets['clean'])

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building %s model', FLAGS.model_name)
    model = efficientnet_builder.build_model(FLAGS.model_name)

    scaled_lr = FLAGS.base_learning_rate * (batch_size / 256.0)
    # Decay epoch is 2.4, warmup epoch is 5 according to the Efficientnet paper.
    decay_steps = steps_per_epoch * 2.4
    warmup_step = steps_per_epoch * 5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        scaled_lr, decay_steps, decay_rate=0.97, staircase=True)
    learning_rate = utils.WarmupDecaySchedule(lr_schedule, warmup_step)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate, rho=0.9, momentum=0.9, epsilon=0.001)

    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
    }
    logging.info('Finished building %s model', FLAGS.model_name)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  def train_step(inputs):
    """Build `step_fn` for efficientnet learning."""
    images, labels = inputs

    num_replicas = tf.cast(strategy.num_replicas_in_sync, tf.float32)
    l2_coeff = tf.cast(FLAGS.l2, tf.float32)

    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      logits = tf.cast(logits, tf.float32)
      # TODO(ywenxu): For better numerical stability, use logits to calculate
      # the loss without softmax layer.
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(
              labels, probs, label_smoothing=FLAGS.label_smoothing))

      def _is_batch_norm(v):
        """Decide whether a variable belongs to `batch_norm`."""
        keywords = ['batchnorm', 'batch_norm', 'bn']
        return any([k in v.name.lower() for k in keywords])

      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights
                          if not _is_batch_norm(v)])
      loss = negative_log_likelihood + l2_coeff * l2_loss
      scaled_loss = loss / num_replicas

    gradients = tape.gradient(scaled_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    metrics['train/loss'].update_state(loss)
    metrics['train/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['train/accuracy'].update_state(labels, probs)

    step_info = {
        'loss/negative_log_likelihood': negative_log_likelihood / num_replicas,
        'loss/total_loss': scaled_loss,
    }
    return step_info

  def eval_step(inputs):
    """A single step."""
    images, labels = inputs
    logits = model(images, training=False)
    logits = tf.cast(logits, tf.float32)
    probs = tf.nn.softmax(logits)
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(labels, probs))
    metrics['test/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['test/accuracy'].update_state(labels, probs)

  @tf.function
  def epoch_fn(should_eval):
    """Build `epoch_fn` for training and potential eval."""
    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      info = strategy.run(train_step, args=(next(train_iterator),))

      optim_step = optimizer.iterations
      if optim_step % tf.cast(100, optim_step.dtype) == 0:
        for k, v in info.items():
          v_reduce = strategy.reduce(tf.distribute.ReduceOp.SUM, v, None)
          tf.summary.scalar(k, v_reduce, optim_step)
        tf.summary.scalar('loss/lr', learning_rate(optim_step), optim_step)
        summary_writer.flush()

    if should_eval:
      for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
        strategy.run(eval_step, args=(next(test_iterator),))

  # Main training loop.
  start_time = time.time()
  with summary_writer.as_default():
    for epoch in range(initial_epoch, FLAGS.train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      should_eval = (epoch % FLAGS.evaluation_interval == 0)
      # Pass tf constant to avoid re-tracing.
      epoch_fn(tf.constant(should_eval))

      current_step = (epoch + 1) * steps_per_epoch
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
      logging.info(message)

      logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                   metrics['train/loss'].result(),
                   metrics['train/accuracy'].result() * 100)

      if should_eval:
        logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                     metrics['test/negative_log_likelihood'].result(),
                     metrics['test/accuracy'].result() * 100)

      total_metrics = metrics.copy()
      total_results = {name: metric.result()
                       for name, metric in total_metrics.items()}
      total_results.update({'lr': learning_rate(optimizer.iterations)})
      with summary_writer.as_default():
        for name, result in total_results.items():
          if should_eval or 'test' not in name:
            tf.summary.scalar(name, result, step=epoch + 1)

      for metric in metrics.values():
        metric.reset_states()

      if (FLAGS.checkpoint_interval > 0 and
          (epoch + 1) % FLAGS.checkpoint_interval == 0):
        checkpoint_name = checkpoint.save(os.path.join(
            FLAGS.output_dir, 'checkpoint'))
        logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  app.run(main)
