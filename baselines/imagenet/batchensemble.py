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

import contextlib
import os
import time

from absl import app
from absl import flags
from absl import logging

import batchensemble_model  # local file import
import utils  # local file import

import tensorflow.compat.v2 as tf

# Flags of batch ensemble model.
flags.DEFINE_integer(
    'num_models', default=1, help='Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_float(
    'random_sign_init', default=1.0,
    help='Use random sign init for fast weights.')
flags.DEFINE_integer('seed', default=0, help='random seeds.')
flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help='Base learning rate when train batch size is 128.')
flags.DEFINE_bool(
    'version2', default=False, help='Use ensemble version2.')
flags.DEFINE_float(
    'weight_decay', default=1e-4, help='weight decay coefficient.')
flags.DEFINE_float(
    'fast_weight_lr_multiplier',
    default=1.0, help='fast weights lr multiplier.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('model_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS

# Imagenet training and test data sets.
# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAINING_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000

# Training hyperparameters.
_EPOCHS = 90

# Allow overriding epochs, steps_per_epoch for testing
flags.DEFINE_integer('num_epochs', _EPOCHS, '')
flags.DEFINE_integer(
    'steps_per_epoch', None,
    'Steps for epoch during training. If unspecified, use default value.')

# Learning rate schedule
_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


class ResnetLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate, num_epochs):
    super(ResnetLearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = _LR_SCHEDULE[0]
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in _LR_SCHEDULE:
      start_epoch = (start_epoch * self.num_epochs) // 90
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate
    }


def safe_mean(losses):
  total = tf.reduce_sum(losses)
  num_elements = tf.dtypes.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)


def main(argv):
  del argv  # unused arg
  tf.enable_v2_behavior()
  tf.random.set_seed(FLAGS.seed)
  if FLAGS.use_gpu:
    device = contextlib.suppress()
  else:
    num_workers = 1
    job_name = 'worker'
    primary_cpu_task = '/job:%s' % job_name

  is_tpu_pod = not FLAGS.use_gpu and num_workers > 1
  # Running BatchEnsemble on tpu pod leads to BatchEnsemble version 2, where the
  # input images are not only tiled in the inference mode but also tiled in the
  # training. BatchEnsemble version 2 means each ensemble member is trained with
  # the same batch size as single model.
  if is_tpu_pod and FLAGS.version2:
    logging.info('Training BatchEnsemble version 2 on tpu pod!')
    batch_size = ((FLAGS.per_core_batch_size // FLAGS.num_models) *
                  FLAGS.num_cores)
  else:
    batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  model_dir = FLAGS.model_dir

  steps_per_epoch = FLAGS.steps_per_epoch
  if not steps_per_epoch:
    steps_per_epoch = APPROX_IMAGENET_TRAINING_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

  logging.info('Saving checkpoints at %s', model_dir)

  if FLAGS.use_gpu:
    logging.info('Use GPU.')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, job_name=job_name)
    # TODO(trandustin): Add infra to insert this on open-source code only.
    # copybara:insert tf.config.experimental_connect_to_host(resolver.master())  # pylint: disable=line-too-long
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  with device:
    # TODO(b/130307853): In TPU Pod, we have to use
    # `strategy.experimental_distribute_datasets_from_function` instead of
    # `strategy.experimental_distribute_dataset` because dataset cannot be
    # cloned in eager mode. And when using
    # `strategy.experimental_distribute_datasets_from_function`, we should use
    # per core batch size instead of global batch size, because no re-batch is
    # happening in this case.
    if is_tpu_pod:
      imagenet_train = utils.ImageNetInput(
          is_training=True,
          data_dir=FLAGS.data_dir,
          batch_size=FLAGS.per_core_batch_size // FLAGS.num_models,
          use_bfloat16=not FLAGS.use_gpu,
          drop_remainder=True)
      imagenet_eval = utils.ImageNetInput(
          is_training=False,
          data_dir=FLAGS.data_dir,
          batch_size=FLAGS.per_core_batch_size // FLAGS.num_models,
          use_bfloat16=not FLAGS.use_gpu,
          drop_remainder=True)
      train_dataset = strategy.experimental_distribute_datasets_from_function(
          imagenet_train.input_fn)
      test_dataset = strategy.experimental_distribute_datasets_from_function(
          imagenet_eval.input_fn)
    else:
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

    with strategy.scope():
      logging.info('Building Keras ResNet-50 model')
      model = batchensemble_model.ensemble_resnet50(
          num_classes=NUM_CLASSES,
          num_models=FLAGS.num_models,
          random_sign_init=FLAGS.random_sign_init)
      base_lr = FLAGS.base_learning_rate * batch_size / 256
      learning_rate = ResnetLearningRateSchedule(steps_per_epoch,
                                                 base_lr,
                                                 FLAGS.num_epochs)
      optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                          momentum=0.9,
                                          nesterov=True)
      training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
      training_celoss = tf.keras.metrics.Mean('training_celoss',
                                              dtype=tf.float32)
      training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'training_accuracy', dtype=tf.float32)
      test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)
      if FLAGS.num_models > 1:
        test_losses = []
        test_accs = []
        for i in range(FLAGS.num_models):
          test_losses.append(
              tf.keras.metrics.Mean('test_loss_{}'.format(i), dtype=tf.float32))
          test_accs.append(tf.keras.metrics.SparseCategoricalAccuracy(
              'test_acc_{}'.format(i), dtype=tf.float32))
      logging.info('Finished building Keras ResNet-50 model')

      checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
      latest_checkpoint = tf.train.latest_checkpoint(model_dir)
      initial_epoch = 0
      if latest_checkpoint:
        # checkpoint.restore must be within a strategy.scope() so that optimizer
        # slot variables are mirrored.
        checkpoint.restore(latest_checkpoint)
        logging.info('Loaded checkpoint %s', latest_checkpoint)
        initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

    # Create summary writers
    summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, 'summaries'))

    @tf.function
    def train_step(iterator):
      """Training StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs

        if FLAGS.version2 and FLAGS.num_models > 1:
          images = tf.tile(images, [FLAGS.num_models, 1, 1, 1])
          labels = tf.tile(labels, [FLAGS.num_models, 1])

        with tf.GradientTape() as tape:
          predictions = model(images, training=True)

          # Loss calculations.
          # Part 1: Prediction loss.
          prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, predictions)
          loss1 = tf.reduce_mean(prediction_loss)
          # Part 2: L2 loss
          filtered_variables = []
          for var in model.trainable_variables:
            if ('bn' not in var.name or 'batch_norm' not in var.name) and (
                'alpha' not in var.name) and ('gamma' not in var.name):
              filtered_variables.append(tf.reshape(var, (-1,)))
          l2_loss = FLAGS.weight_decay * 2 * tf.nn.l2_loss(
              tf.concat(filtered_variables, axis=0))

          # Scale the loss given the TPUStrategy will reduce sum all gradients.
          loss = loss1 + l2_loss
          scaled_loss = loss / strategy.num_replicas_in_sync

        grads = tape.gradient(scaled_loss, model.trainable_variables)

        # Separate learning rate implementation.
        grad_list = []
        if FLAGS.fast_weight_lr_multiplier != 1.0:
          grads_and_vars = list(zip(grads, model.trainable_variables))
          for vec, var in grads_and_vars:
            if 'bn' not in var.name or 'batch_norm' not in var.name:
              if ('alpha' in var.name) or ('gamma' in var.name) or (
                  'bias' in var.name):
                grad_list.append((vec * FLAGS.fast_weight_lr_multiplier, var))
              else:
                grad_list.append((vec, var))
            else:
              grad_list.append((vec, var))
          optimizer.apply_gradients(grad_list)
        else:
          optimizer.apply_gradients(zip(grads, model.trainable_variables))

        training_loss.update_state(loss)
        training_celoss.update_state(loss1)
        training_accuracy.update_state(labels, predictions)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    @tf.function
    def test_step(iterator):
      """Evaluation StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        if FLAGS.num_models > 0:
          images = tf.tile(images, [FLAGS.num_models, 1, 1, 1])
        predictions = model(images, training=False)

        if FLAGS.num_models > 1:
          per_predictions = tf.split(
              predictions, num_or_size_splits=FLAGS.num_models, axis=0)
          for i in range(FLAGS.num_models):
            member_prediction = per_predictions[i]
            member_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, member_prediction)
            test_losses[i].update_state(member_loss)
            test_accs[i].update_state(labels, member_prediction)

          ensemble_prediction = tf.add_n(per_predictions) / FLAGS.num_models
          loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, ensemble_prediction)
          loss = safe_mean(loss)
          test_loss.update_state(loss)
          test_accuracy.update_state(labels, ensemble_prediction)
        else:
          loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, predictions)
          loss = safe_mean(loss)
          test_loss.update_state(loss)
          test_accuracy.update_state(labels, predictions)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    train_iterator = iter(train_dataset)
    work_unit = xm_client.get_current_work_unit()
    start_time = time.time()

    for epoch in range(initial_epoch, FLAGS.num_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      with summary_writer.as_default():
        for step in range(steps_per_epoch):
          if step % 20 == 0:
            logging.info('Running step %s in epoch %s', step, epoch)
          train_step(train_iterator)
          current_step = epoch * steps_per_epoch + step
          max_steps = steps_per_epoch * FLAGS.num_epochs
          time_elapsed = time.time() - start_time
          steps_per_sec = float(current_step) / time_elapsed
          # TODO(trandustin): Add this best practice to other scripts.
          eta_seconds = (max_steps - current_step) / (steps_per_sec + 1e-7)
          message = ('{:.1f}% completion, at step {:d}. {:.1f} steps/s. '
                     'ETA: {:.0f} min'.format(100 * current_step / max_steps,
                                              current_step,
                                              steps_per_sec,
                                              eta_seconds / 60))
          work_unit.set_notes(message)

        tf.summary.scalar('train/loss',
                          training_loss.result(),
                          step=optimizer.iterations)
        tf.summary.scalar('train/ce_loss',
                          training_celoss.result(),
                          step=optimizer.iterations)
        tf.summary.scalar('train/accuracy',
                          training_accuracy.result(),
                          step=optimizer.iterations)
        logging.info('Training loss: %s, accuracy: %s%%',
                     round(float(training_loss.result()), 4),
                     round(float(training_accuracy.result() * 100), 2))

        current_step = (epoch + 1) * steps_per_epoch
        result_dict = {
            'train_celoss': training_celoss.result(),
            'train_acc': training_accuracy.result(),
            'train_loss': training_loss.result()
        }
        for name in ['train_celoss', 'train_acc', 'train_loss']:
          objective = work_unit.get_measurement_series(label=name)
          objective.create_measurement(result_dict[name], current_step)

        training_loss.reset_states()
        training_celoss.reset_states()
        training_accuracy.reset_states()

        test_iterator = iter(test_dataset)
        for step in range(steps_per_eval):
          if step % 20 == 0:
            logging.info('Starting to run eval step %s of epoch: %s', step,
                         epoch)
          test_step(test_iterator)
        tf.summary.scalar('test/loss',
                          test_loss.result(),
                          step=optimizer.iterations)
        tf.summary.scalar('test/accuracy',
                          test_accuracy.result(),
                          step=optimizer.iterations)
        logging.info('Test loss: %s, accuracy: %s%%',
                     round(float(test_loss.result()), 4),
                     round(float(test_accuracy.result() * 100), 2))

        current_step = (epoch + 1) * steps_per_epoch
        result_dict = {
            'test_acc': test_accuracy.result(),
            'test_loss': test_loss.result()
        }
        for name in ['test_acc', 'test_loss']:
          objective = work_unit.get_measurement_series(label=name)
          objective.create_measurement(result_dict[name], current_step)

        test_loss.reset_states()
        test_accuracy.reset_states()

        if FLAGS.num_models > 1:
          for i in range(FLAGS.num_models):
            tf.summary.scalar('test/ensemble_loss_member{}'.format(i),
                              test_losses[i].result(),
                              step=optimizer.iterations)
            tf.summary.scalar('test/ensemble_accuracy_member{}'.format(i),
                              test_accs[i].result(),
                              step=optimizer.iterations)
            logging.info('Member %d Test loss: %s, accuracy: %s%%',
                         i, round(float(test_losses[i].result()), 4),
                         round(float(test_accs[i].result() * 100), 2))
            test_losses[i].reset_states()
            test_accs[i].reset_states()

      checkpoint_name = checkpoint.save(os.path.join(model_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
