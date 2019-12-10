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

"""ResNet-32x4 on CIFAR-10 trained with maximum likelihood and gradient descent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from absl import app
from absl import flags
from absl import logging

import utils  # local file import
import six
from six.moves import range

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset: cifar10 or cifar100.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores.')
FLAGS = flags.FLAGS

_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 1), (0.1, 80), (0.01, 160), (0.001, 180)
]


def resnet_layer(inputs,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 l2=0.):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    kernel_size: Kernel dimensions for Conv2D.
    strides: Stride dimensinons for Conv2D.
    activation: tf.keras.activations.Activation.
    l2: L2 regularization coefficient.

  Returns:
    tf.Tensor.
  """
  x = inputs
  logging.info('Applying conv layer.')
  x = tf.keras.layers.Conv2D(
      filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same',
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.BatchNormalization(epsilon=1e-5,
                                         momentum=0.9)(x)
  if activation is not None:
    x = tf.keras.layers.Activation(activation)(x)
  return x


def resnet_v1(input_shape, depth, width_multiplier, num_classes, l2):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    width_multiplier: Integer to multiply the number of typical filters by.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.

  Returns:
    tf.keras.Model.
  """
  num_res_blocks = (depth - 2) // 6
  filters = 16 * width_multiplier
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = resnet_layer(inputs,
                   filters=filters,
                   activation='relu',
                   l2=l2)
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = resnet_layer(x,
                       filters=filters,
                       strides=strides,
                       activation='relu',
                       l2=l2)
      y = resnet_layer(y,
                       filters=filters,
                       activation=None,
                       l2=l2)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(x,
                         filters=filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         l2=l2)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2))(x)
  return tf.keras.models.Model(inputs=inputs, outputs=x)


def main(argv):
  del argv  # unused arg
  if FLAGS.use_gpu:
    raise ValueError('Only TPU is currently supported.')

  tf.enable_v2_behavior()
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)
  job_name = 'worker'
  primary_cpu_task = '/job:%s' % job_name

  logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else 'local')
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu, job_name=job_name)
  # copybara:insert tf.config.experimental_connect_to_host(resolver.master())  # pylint: disable=line-too-long
  tf.tpu.experimental.initialize_tpu_system(resolver)
  strategy = tf.distribute.experimental.TPUStrategy(resolver)

  with tf.device(primary_cpu_task):
    def train_input_fn(ctx):
      """Sets up local (per-core) dataset batching."""
      dataset = utils.load_distributed_dataset(
          split=tfds.Split.TRAIN,
          name=FLAGS.dataset,
          batch_size=FLAGS.per_core_batch_size,
          drop_remainder=True,
          use_bfloat16=True)
      if ctx and ctx.num_input_pipelines > 1:
        dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
      return dataset

    def test_input_fn(ctx):
      """Sets up local (per-core) dataset batching."""
      dataset = utils.load_distributed_dataset(
          split=tfds.Split.TEST,
          name=FLAGS.dataset,
          batch_size=FLAGS.per_core_batch_size,
          drop_remainder=True,
          use_bfloat16=True)
      if ctx and ctx.num_input_pipelines > 1:
        dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
      return dataset

    train_dataset = strategy.experimental_distribute_datasets_from_function(
        train_input_fn)
    test_dataset = strategy.experimental_distribute_datasets_from_function(
        test_input_fn)
    ds_info = tfds.builder(FLAGS.dataset).info

    batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
    steps_per_eval = ds_info.splits['test'].num_examples // batch_size

    with strategy.scope():
      logging.info('Building Keras ResNet-32 model')
      model = resnet_v1(input_shape=ds_info.features['image'].shape,
                        depth=32,
                        num_classes=ds_info.features['label'].num_classes,
                        width_multiplier=4,
                        l2=FLAGS.l2)
      logging.info('Model input shape: %s', model.input_shape)
      logging.info('Model output shape: %s', model.output_shape)
      logging.info('Model number of weights: %s', model.count_params())
      base_lr = FLAGS.base_learning_rate * batch_size / 128
      lr_schedule = utils.ResnetLearningRateSchedule(steps_per_epoch,
                                                     base_lr,
                                                     _LR_SCHEDULE)
      optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                          momentum=0.9,
                                          nesterov=True)
      train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
      train_nll = tf.keras.metrics.Mean('train_nll', dtype=tf.float32)
      train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'train_accuracy', dtype=tf.float32)
      test_nll = tf.keras.metrics.Mean('test_nll', dtype=tf.float32)
      test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
          'test_accuracy', dtype=tf.float32)

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
        os.path.join(FLAGS.output_dir, 'summaries/'))

    @tf.function
    def train_step(iterator):
      """Training StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        with tf.GradientTape() as tape:
          logits = model(images, training=True)
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))
          l2_loss = sum(model.losses)
          loss = negative_log_likelihood + l2_loss
          # Scale the loss given the TPUStrategy will reduce sum all gradients.
          scaled_loss = loss / strategy.num_replicas_in_sync

        grads = tape.gradient(scaled_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss.update_state(loss)
        train_nll.update_state(negative_log_likelihood)
        train_accuracy.update_state(labels, logits)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    @tf.function
    def test_step(iterator):
      """Evaluation StepFn."""
      def step_fn(inputs):
        """Per-Replica StepFn."""
        images, labels = inputs
        logits = model(images, training=False)
        probs = tf.nn.softmax(logits)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, probs))
        test_nll.update_state(negative_log_likelihood)
        test_accuracy.update_state(labels, probs)

      strategy.experimental_run_v2(step_fn, args=(next(iterator),))

    train_iterator = iter(train_dataset)

    start_time = time.time()

    for epoch in range(initial_epoch, FLAGS.train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      with summary_writer.as_default():
        for step in range(steps_per_epoch):
          if step % 20 == 0:
            logging.info('Running step %s in epoch %s', step, epoch)
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
          work_unit.set_notes(message)

        tf.summary.scalar('train/loss',
                          train_loss.result(),
                          step=epoch + 1)
        tf.summary.scalar('train/negative_log_likelihood',
                          train_nll.result(),
                          step=epoch + 1)
        tf.summary.scalar('train/accuracy',
                          train_accuracy.result(),
                          step=epoch + 1)
        logging.info('Train Loss: %s, Accuracy: %s%%',
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

      if (epoch + 1) % 20 == 0:
        checkpoint_name = checkpoint.save(
            os.path.join(FLAGS.output_dir, 'checkpoint'))
        logging.info('Saved checkpoint to %s', checkpoint_name)

if __name__ == '__main__':
  app.run(main)
