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

"""Batch Ensemble ResNet-50."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

import batchensemble_model  # local file import
import utils  # local file import

import six
import tensorflow.compat.v2 as tf

flags.DEFINE_integer('num_models', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_bool('version2', True, 'Use ensemble version2.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_float('fast_weight_lr_multiplier', 2.5,
                   'fast weights lr multiplier.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 100, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000

_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


# TODO(trandustin): Replace with logits similar to CIFAR code.
def safe_mean(losses):
  total = tf.reduce_sum(losses)
  num_elements = tf.dtypes.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)


def main(argv):
  del argv  # unused arg
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)

  # In BatchEnsemble version 2, the
  # input images are not only tiled in the inference mode but also tiled in the
  # training. BatchEnsemble version 2 means each ensemble member is trained with
  # the same batch size as single model.
  if FLAGS.version2:
    logging.info('Training BatchEnsemble version 2')
    batch_size = ((FLAGS.per_core_batch_size // FLAGS.num_models) *
                  FLAGS.num_cores)
  else:
    batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

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

  imagenet_train = utils.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      use_bfloat16=not FLAGS.use_gpu,
      drop_remainder=True)
  imagenet_eval = utils.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=batch_size,
      use_bfloat16=not FLAGS.use_gpu,
      drop_remainder=True)
  train_dataset = strategy.experimental_distribute_dataset(
      imagenet_train.input_fn())
  test_dataset = strategy.experimental_distribute_dataset(
      imagenet_eval.input_fn())

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = batchensemble_model.ensemble_resnet50(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES,
        num_models=FLAGS.num_models,
        random_sign_init=FLAGS.random_sign_init,
        use_tpu=not FLAGS.use_gpu)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 256
    learning_rate = utils.LearningRateSchedule(steps_per_epoch,
                                               base_lr,
                                               FLAGS.train_epochs,
                                               _LR_SCHEDULE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        momentum=0.9,
                                        nesterov=True)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_nll = tf.keras.metrics.Mean('train_nll', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    test_nll = tf.keras.metrics.Mean('test_nll', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    test_nlls = []
    test_accs = []
    for i in range(FLAGS.num_models):
      test_nlls.append(
          tf.keras.metrics.Mean('test_nll_{}'.format(i), dtype=tf.float32))
      test_accs.append(tf.keras.metrics.SparseCategoricalAccuracy(
          'test_acc_{}'.format(i), dtype=tf.float32))
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

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs

      if FLAGS.version2:
        images = tf.tile(images, [FLAGS.num_models, 1, 1, 1])
        labels = tf.tile(labels, [FLAGS.num_models, 1])

      with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        if FLAGS.use_bfloat16:
          predictions = tf.cast(predictions, tf.float32)

        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions)
        loss1 = tf.reduce_mean(prediction_loss)
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the slow weights and bias terms. This excludes BN
          # parameters and fast weight approximate posterior/prior parameters,
          # but pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = loss1 + l2_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weight approximate
          # posterior/prior parameters. This is excludes BN and slow weights,
          # but pay caution to the naming scheme.
          if (('bn' not in var.name or 'batch_norm' not in var.name) and
              'kernel' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier,
                                   var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      train_loss.update_state(loss)
      train_nll.update_state(loss1)
      train_accuracy.update_state(labels, predictions)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      images = tf.tile(images, [FLAGS.num_models, 1, 1, 1])
      predictions = model(images, training=False)
      if FLAGS.use_bfloat16:
        predictions = tf.cast(predictions, tf.float32)

      per_predictions = tf.split(
          predictions, num_or_size_splits=FLAGS.num_models, axis=0)
      for i in range(FLAGS.num_models):
        member_prediction = per_predictions[i]
        member_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, member_prediction)
        test_nlls[i].update_state(member_loss)
        test_accs[i].update_state(labels, member_prediction)

      ensemble_prediction = tf.add_n(per_predictions) / FLAGS.num_models
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, ensemble_prediction)
      loss = safe_mean(loss)
      test_nll.update_state(loss)
      test_accuracy.update_state(labels, ensemble_prediction)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    with summary_writer.as_default():
      for step in range(steps_per_epoch):
        train_step(train_iterator)

        current_step = epoch * steps_per_epoch + step
        max_steps = steps_per_epoch * FLAGS.train_epochs
        time_elapsed = time.time() - start_time
        steps_per_sec = float(current_step) / time_elapsed
        eta_seconds = (max_steps - current_step) / (steps_per_sec + 1e-7)
        message = ('{:.1f}% completion, at step {:d}. {:.1f} steps/s. '
                   'ETA: {:.0f} min'.format(100 * current_step / max_steps,
                                            current_step,
                                            steps_per_sec,
                                            eta_seconds / 60))
        if step % 20 == 0:
          logging.info(message)

      tf.summary.scalar('train/loss',
                        train_loss.result(),
                        step=epoch + 1)
      tf.summary.scalar('train/nll',
                        train_nll.result(),
                        step=epoch + 1)
      tf.summary.scalar('train/accuracy',
                        train_accuracy.result(),
                        step=epoch + 1)
      logging.info('Train loss: %s, Accuracy: %s%%',
                   round(float(train_loss.result()), 4),
                   round(float(train_accuracy.result() * 100), 2))

      train_loss.reset_states()
      train_nll.reset_states()
      train_accuracy.reset_states()

      test_iterator = iter(test_dataset)
      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        test_step(test_iterator)
      tf.summary.scalar('test/negative_log_likelihood',
                        test_nll.result(),
                        step=epoch + 1)
      tf.summary.scalar('test/accuracy',
                        test_accuracy.result(),
                        step=epoch + 1)
      logging.info('Test NLL: %s, Accuracy: %s%%',
                   round(float(test_nll.result()), 4),
                   round(float(test_accuracy.result() * 100), 2))

      test_nll.reset_states()
      test_accuracy.reset_states()

      for i in range(FLAGS.num_models):
        tf.summary.scalar('test/ensemble_nll_member{}'.format(i),
                          test_nlls[i].result(),
                          step=epoch + 1)
        tf.summary.scalar('test/ensemble_accuracy_member{}'.format(i),
                          test_accs[i].result(),
                          step=epoch + 1)
        logging.info('Member %d Test loss: %s, accuracy: %s%%',
                     i, round(float(test_nlls[i].result()), 4),
                     round(float(test_accs[i].result() * 100), 2))
        test_nlls[i].reset_states()
        test_accs[i].reset_states()

    checkpoint_name = checkpoint.save(os.path.join(
        FLAGS.output_dir, 'checkpoint'))
    logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  app.run(main)
