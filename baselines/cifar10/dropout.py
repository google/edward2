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

"""ResNet-20 on CIFAR-10 with Monte Carlo dropout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import app
from absl import flags
from absl import logging
import utils  # local file import

from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('output_dir', '/tmp/cifar10', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_float('init_learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_float('l2', 3e-4, 'L2 regularization coefficient.')
flags.DEFINE_float('dropout_rate', 0.05, 'Dropout rate.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


def resnet_layer(inputs,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 l2=0.,
                 dropout_rate=0.):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    kernel_size: Kernel dimensions for Conv2D.
    strides: Stride dimensinons for Conv2D.
    activation: tf.keras.activations.Activation.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.

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
      kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Dropout(dropout_rate)(x, training=True)
  x = tf.keras.layers.BatchNormalization()(x)
  if activation is not None:
    x = tf.keras.layers.Activation(activation)(x)
  return x


def resnet_v1(input_shape, depth, num_classes, l2, dropout_rate):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.

  Returns:
    tf.keras.Model.
  """
  num_res_blocks = (depth - 2) // 6
  filters = 16
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')
  layer = functools.partial(resnet_layer,
                            l2=l2,
                            dropout_rate=dropout_rate)

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = layer(inputs,
            filters=filters,
            activation='relu')
  for stack in range(3):
    for res_block in range(num_res_blocks):
      logging.info('Starting ResNet stack #%d block #%d.', stack, res_block)
      strides = 1
      if stack > 0 and res_block == 0:  # first layer but not first stack
        strides = 2  # downsample
      y = layer(x,
                filters=filters,
                strides=strides,
                activation='relu')
      y = layer(y,
                filters=filters,
                activation=None)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = layer(x,
                  filters=filters,
                  kernel_size=1,
                  strides=strides,
                  activation=None)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Dropout(dropout_rate)(x, training=True)
  return tf.keras.models.Model(inputs=inputs, outputs=x)


def main(argv):
  del argv  # unused arg
  if FLAGS.num_cores > 1 or not FLAGS.use_gpu:
    raise ValueError('Only single GPU is currently supported.')
  tf.enable_v2_behavior()
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  dataset_train, ds_info = utils.load_dataset(tfds.Split.TRAIN, with_info=True)
  dataset_size = ds_info.splits['train'].num_examples
  dataset_train = dataset_train.repeat().shuffle(10 * FLAGS.batch_size).batch(
      FLAGS.batch_size)
  test_batch_size = 100
  validation_steps = ds_info.splits['test'].num_examples // test_batch_size
  dataset_test = utils.load_dataset(tfds.Split.TEST)
  dataset_test = dataset_test.repeat().batch(test_batch_size)

  model = resnet_v1(input_shape=ds_info.features['image'].shape,
                    depth=20,
                    num_classes=ds_info.features['label'].num_classes,
                    l2=FLAGS.l2,
                    dropout_rate=FLAGS.dropout_rate)
  model.compile(
      tf.keras.optimizers.SGD(FLAGS.init_learning_rate,
                              momentum=0.9,
                              nesterov=True),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[
          tf.keras.metrics.SparseCategoricalCrossentropy(
              name='negative_log_likelihood',
              from_logits=True),
          tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=FLAGS.output_dir,
                                                  write_graph=False)
  lr_scheduler = utils.make_lr_scheduler(FLAGS.init_learning_rate)
  model.fit(dataset_train,
            steps_per_epoch=dataset_size // FLAGS.batch_size,
            epochs=FLAGS.train_epochs,
            validation_data=dataset_test,
            validation_steps=validation_steps,
            callbacks=[tensorboard_cb, lr_scheduler])

  logging.info('Saving model to output_dir.')
  model_filename = FLAGS.output_dir + '/model.ckpt'
  model.save_weights(model_filename)

if __name__ == '__main__':
  app.run(main)
