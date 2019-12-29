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

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset: cifar10 or cifar100.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
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
  tf.enable_v2_behavior()
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
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  def train_input_fn(ctx):
    """Sets up local (per-core) dataset batching."""
    dataset = utils.load_distributed_dataset(
        split=tfds.Split.TRAIN,
        name=FLAGS.dataset,
        batch_size=FLAGS.per_core_batch_size,
        drop_remainder=True,
        use_bfloat16=FLAGS.use_bfloat16)
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
        use_bfloat16=FLAGS.use_bfloat16)
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

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

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
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
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
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
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

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, probs))
      metrics['test/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['test/accuracy'].update_state(labels, probs)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

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
      if step % 20 == 0:
        logging.info(message)

    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      if step % 20 == 0:
        logging.info('Starting to run eval step %s of epoch: %s', step, epoch)
      test_step(test_iterator)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    with summary_writer.as_default():
      for name, metric in metrics.items():
        tf.summary.scalar(name, metric.result(), step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (epoch + 1) % 20 == 0:
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

if __name__ == '__main__':
  app.run(main)
