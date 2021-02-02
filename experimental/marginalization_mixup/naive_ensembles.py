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

"""Wide ResNet 28-10 on CIFAR-10/100 trained with maximum likelihood.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.
"""

import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

from experimental.marginalization_mixup import data_utils  # local file import
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.baselines.cifar import utils
import uncertainty_metrics as um

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('ensemble_size', 4, 'Ensemble size of models.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', 200,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('cifar_class', None,
                     'Specialized class. None means train on all classes.')
flags.DEFINE_bool('random_augment', False,
                  'Whether random augment the input data.')
flags.DEFINE_bool('augmix', False, 'Whether augmix the input data.')
flags.DEFINE_integer('aug_count', 1, 'How many augmentations on input image.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth.')
flags.DEFINE_integer('augmix_width', 3, 'Augmix width')
flags.DEFINE_bool('distance_logits', False, 'whether use distance logits.')
flags.DEFINE_bool('one_vs_all', False, 'whether use one vs all distance loss.')
flags.DEFINE_bool('adaptive_mixup', False, 'whether use adaptive mixup.')
flags.DEFINE_float('mixup_alpha', -1., 'Mixup hyperparameter, 0. to diable.')
flags.DEFINE_float('adaptive_lr', 1, 'Mixup adapative lr.')
flags.DEFINE_bool('validation', False, 'Whether to use validation set.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

BatchNormalization = functools.partial(  # pylint: disable=invalid-name
    tf.python.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)
Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.python.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


class DistanceMax(tf.python.keras.layers.Layer):
  r"""Implements the output layer of model for Distinction Maximization Loss.

  In Distinction Maximization loss, the logits produced by the output layer of
  a neural network are defined as `logits = - ||f_{\theta}(x) - W||`/. This
  layer implements the loss as specified here - https://arxiv.org/abs/1908.05569
  """

  def __init__(self, num_classes=10):
    super(DistanceMax, self).__init__()
    self.num_classes = num_classes

  def build(self, input_shape):
    self.w = self.add_weight('w',
                             shape=(input_shape[-1], self.num_classes),
                             initializer='zeros',
                             trainable=True)

  def call(self, inputs):
    distances = tf.norm(
        tf.expand_dims(inputs, axis=-1) - tf.expand_dims(self.w, axis=0),
        axis=1)
    # In DM Loss, the probability predictions do not have the alpha term.
    return -1.0 * distances


class NaiveEnsemble(tf.python.keras.Model):
  """A keras model wrapper for naive ensembles."""

  def __init__(self, models, output_dirs=None):
    super(NaiveEnsemble, self).__init__()
    self.models = models
    self.ensemble_size = len(models)
    if output_dirs is not None:
      for i, output_dir in enumerate(output_dirs):
        checkpoint = tf.train.Checkpoint(model=self.models[i])
        latest_checkpoint = tf.train.latest_checkpoint(output_dir)
        if latest_checkpoint:
          checkpoint.restore(latest_checkpoint)
          logging.info('Loaded checkpoint %s', latest_checkpoint)

  def call(self,
           inputs,
           split_inputs=False,
           training=True):
    outputs = []
    if split_inputs:
      inputs = tf.split(inputs, self.ensemble_size, axis=0)
      for i in range(self.ensemble_size):
        outputs.append(self.models[i](inputs[i], training=training))
    else:
      for i in range(self.ensemble_size):
        outputs.append(self.models[i](inputs, training=training))
    return tf.concat(outputs, axis=0)


def basic_block(inputs, filters, strides, l2, version):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  if version == 2:
    y = BatchNormalization(beta_regularizer=tf.python.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.python.keras.regularizers.l2(l2))(y)
    y = tf.python.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=strides,
             kernel_regularizer=tf.python.keras.regularizers.l2(l2))(y)
  y = BatchNormalization(beta_regularizer=tf.python.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.python.keras.regularizers.l2(l2))(y)
  y = tf.python.keras.layers.Activation('relu')(y)
  y = Conv2D(filters,
             strides=1,
             kernel_regularizer=tf.python.keras.regularizers.l2(l2))(y)
  if version == 1:
    y = BatchNormalization(beta_regularizer=tf.python.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.python.keras.regularizers.l2(l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=tf.python.keras.regularizers.l2(l2))(x)
  x = tf.python.keras.layers.add([x, y])
  if version == 1:
    x = tf.python.keras.layers.Activation('relu')(x)
  return x


def group(inputs, filters, strides, num_blocks, l2, version):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, l2=l2,
                  version=version)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, l2=l2, version=version)
  return x


def wide_resnet(input_shape, depth, width_multiplier, num_classes,
                l2, version, distance_logits=False):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    version: 1, indicating the original ordering from He et al. (2015); or 2,
      indicating the preactivation ordering from He et al. (2016).
    distance_logits: Bool, whether to use distance logits.

  Returns:
    tf.python.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.python.keras.layers.Input(shape=input_shape)
  x = Conv2D(16,
             strides=1,
             kernel_regularizer=tf.python.keras.regularizers.l2(l2))(inputs)
  if version == 1:
    x = BatchNormalization(beta_regularizer=tf.python.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.python.keras.regularizers.l2(l2))(x)
    x = tf.python.keras.layers.Activation('relu')(x)
  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            l2=l2,
            version=version)
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            version=version)
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            version=version)
  if version == 2:
    x = BatchNormalization(beta_regularizer=tf.python.keras.regularizers.l2(l2),
                           gamma_regularizer=tf.python.keras.regularizers.l2(l2))(x)
    x = tf.python.keras.layers.Activation('relu')(x)
  x = tf.python.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.python.keras.layers.Flatten()(x)
  if distance_logits:
    x = DistanceMax(num_classes=num_classes)(x)
  else:
    x = tf.python.keras.layers.Dense(
        num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.python.keras.regularizers.l2(l2),
        bias_regularizer=tf.python.keras.regularizers.l2(l2))(x)
  return tf.python.keras.Model(inputs=inputs, outputs=x)


def one_vs_rest_dm_loss(labels, logits, dm_alpha=1.):
  """Implements the one-vs-all distance-based loss function.

  As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
  logits by dm_alpha before taking K independent sigmoid operations of each
  class logit, and then calculating the sum of the log-loss across classes.

  Args:
    labels: Integer Tensor of dense labels, shape [batch_size].
    logits: Tensor of shape [batch_size, num_classes].
    dm_alpha: Interger of distance logits multiplier.

  Returns:
    A scalar containing the mean over the batch for one-vs-all loss.
  """
  # For the loss function, multiply the logits by alpha before normalization.
  eps = 1e-6
  logits = logits * dm_alpha
  n_classes = tf.cast(logits.shape[1], tf.float32)

  one_vs_rest_probs = tf.math.sigmoid(logits)
  labels = tf.cast(tf.squeeze(labels), tf.int32)
  row_ids = tf.range(tf.shape(one_vs_rest_probs)[0], dtype=tf.int32)
  idx = tf.stack([row_ids, labels], axis=1)

  class_probs = tf.gather_nd(one_vs_rest_probs, idx)

  loss = (
      tf.reduce_mean(tf.math.log(class_probs + eps)) +
      n_classes * tf.reduce_mean(tf.math.log(1. - one_vs_rest_probs + eps)) -
      tf.reduce_mean(tf.math.log(1. - class_probs + eps)))

  return -loss


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
      'ensemble_size': FLAGS.ensemble_size,
      'mixup_alpha': FLAGS.mixup_alpha,
      'random_augment': FLAGS.random_augment,
      'adaptive_mixup': FLAGS.adaptive_mixup,
  }
  train_input_fn = data_utils.load_input_fn(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16,
      validation_set=FLAGS.validation,
      aug_params=aug_params)
  if FLAGS.adaptive_mixup:
    validation_input_fn = data_utils.load_input_fn(
        split='validation',
        name=FLAGS.dataset,
        batch_size=FLAGS.per_core_batch_size,
        use_bfloat16=FLAGS.use_bfloat16,
        validation_set=FLAGS.adaptive_mixup)
    val_dataset = strategy.experimental_distribute_datasets_from_function(
        validation_input_fn)
  clean_test_input_fn = data_utils.load_input_fn(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = strategy.experimental_distribute_datasets_from_function(
      train_input_fn)
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
            batch_size=FLAGS.per_core_batch_size * FLAGS.num_cores,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  if FLAGS.adaptive_mixup:
    # TODO(ywenxu): Remove hard-coded 0.95 in the future.
    total_train_examples = ds_info.splits['train'].num_examples
    num_val_examples = int(total_train_examples * 0.95)
    steps_per_epoch = (total_train_examples - num_val_examples) // batch_size
    steps_per_val = num_val_examples // batch_size
  else:
    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes
  if FLAGS.cifar_class is not None:
    num_classes = 2

  if FLAGS.use_bfloat16:
    policy = tf.python.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.python.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')
    models = []
    for _ in range(FLAGS.ensemble_size):
      models.append(wide_resnet(input_shape=ds_info.features['image'].shape,
                                depth=28,
                                width_multiplier=10,
                                num_classes=num_classes,
                                l2=FLAGS.l2,
                                version=2,
                                distance_logits=FLAGS.distance_logits))
    model = NaiveEnsemble(models)
    logging.info('Model input shape: %s', models[0].input_shape)
    logging.info('Model output shape: %s', models[0].output_shape)
    logging.info('Model number of weights: %s', models[0].count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(start_epoch * FLAGS.train_epochs) // 200
                       for start_epoch in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.python.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'train/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.python.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.python.keras.metrics.Mean(),
        'test/accuracy': tf.python.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/member_accuracy_mean': (
            tf.python.keras.metrics.SparseCategoricalAccuracy()),
        'test/member_ece_mean': um.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins)
    }

    test_diversity = {}
    corrupt_diversity = {}
    if FLAGS.ensemble_size > 1:
      test_diversity = {
          'test/disagreement': tf.python.keras.metrics.Mean(),
          'test/average_kl': tf.python.keras.metrics.Mean(),
          'test/cosine_similarity': tf.python.keras.metrics.Mean(),
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
      images, labels = inputs
      if FLAGS.adaptive_mixup:
        images = tf.identity(images)
      elif FLAGS.augmix or FLAGS.random_augment:
        images_shape = tf.shape(images)
        images = tf.reshape(tf.transpose(
            images, [1, 0, 2, 3, 4]), [-1, images_shape[2],
                                       images_shape[3], images_shape[4]])
      else:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      if FLAGS.cifar_class is not None:
        labels = tf.cast(labels == FLAGS.cifar_class, tf.float32)

      if FLAGS.mixup_alpha > 0:
        split_inputs = True
        if not FLAGS.augmix and not FLAGS.adaptive_mixup:
          labels = tf.tile(labels, [FLAGS.ensemble_size, 1])
        else:
          labels = tf.identity(labels)
      else:
        labels = tf.tile(labels, [FLAGS.ensemble_size])
        split_inputs = True if FLAGS.augmix else False

      with tf.GradientTape() as tape:
        logits = model(images, split_inputs, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        if FLAGS.mixup_alpha > 0:
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
        if FLAGS.distance_logits and FLAGS.one_vs_all:
          distance_loss = one_vs_rest_dm_loss(labels, logits)
          loss = distance_loss + l2_loss
        else:
          loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      if FLAGS.mixup_alpha > 0:
        labels = tf.argmax(labels, axis=-1)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      if FLAGS.cifar_class is not None:
        labels = tf.cast(labels == FLAGS.cifar_class, tf.float32)
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)
      per_probs = tf.split(probs,
                           num_or_size_splits=FLAGS.ensemble_size,
                           axis=0)

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
      tiled_labels = tf.tile(labels, [FLAGS.ensemble_size])
      tiled_probs = tf.concat(per_probs, axis=0)
      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
        metrics['test/member_accuracy_mean'].update_state(
            tiled_labels, tiled_probs)
        metrics['test/member_ece_mean'].update_state(tiled_labels, tiled_probs)
      elif dataset_name != 'validation':
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/member_acc_mean_{}'.format(
            dataset_name)].update_state(tiled_labels, tiled_probs)
        corrupt_metrics['test/member_ece_mean_{}'.format(
            dataset_name)].update_state(tiled_labels, tiled_probs)

      if dataset_name == 'validation':
        return per_probs_tensor, labels
    if dataset_name == 'validation':
      return strategy.run(step_fn, args=(next(iterator),))
    else:
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
      if step % 20 == 0:
        logging.info(message)

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
          batch_size=FLAGS.per_core_batch_size,
          use_bfloat16=FLAGS.use_bfloat16,
          aug_params=aug_params,
          validation_set=True)
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

    total_metrics = metrics.copy()
    total_metrics.update(test_diversity)
    total_results = {name: metric.result()
                     for name, metric in total_metrics.items()}
    total_results.update(corrupt_results)
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

    if FLAGS.adaptive_mixup and epoch == (FLAGS.train_epochs - 2):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'last_but_one_checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)

if __name__ == '__main__':
  app.run(main)
