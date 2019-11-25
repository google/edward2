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

"""ResNet-20 on CIFAR-10 trained with maximum likelihood and gradient descent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import edward2 as ed
import utils  # local file import

from six.moves import range
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
# TODO(trandustin): batch size, init_learning_rate are from Ovadia et al.(2019).
# Tune this given changes.
flags.DEFINE_integer('batch_size', 7, 'Batch size.')
flags.DEFINE_float('init_learning_rate', 0.000717, 'Learning rate.')
# TODO(trandustin): Taken from Keras' example.
# flags.DEFINE_integer('batch_size', 32, 'Batch size.')
# flags.DEFINE_float('init_learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_boolean('batch_norm', True, 'Whether to apply batchnorm.')
FLAGS = flags.FLAGS


def resnet_layer(inputs,
                 filters,
                 kernel_size=3,
                 strides=1,
                 activation=None,
                 batch_norm=True):
  """2D Convolution-Batch Normalization-Activation stack builder.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    kernel_size: Kernel dimensions for Conv2D.
    strides: Stride dimensinons for Conv2D.
    activation: tf.keras.activations.Activation.
    batch_norm: Whether to apply batch normalization.

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
      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
  if batch_norm:
    x = tf.keras.layers.BatchNormalization()(x)
  if activation is not None:
    x = tf.keras.layers.Activation(activation)(x)
  return x


def resnet_v1(input_shape,
              depth,
              num_classes,
              batch_norm):
  """Builds ResNet v1.

  Args:
    input_shape: tf.Tensor.
    depth: ResNet depth.
    num_classes: Number of output classes.
    batch_norm: Whether to apply batch normalization.

  Returns:
    tf.keras.Model.
  """
  num_res_blocks = (depth - 2) // 6
  filters = 16
  if (depth - 2) % 6 != 0:
    raise ValueError('depth must be 6n+2 (e.g. 20, 32, 44).')

  logging.info('Starting ResNet build.')
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = resnet_layer(inputs,
                   filters=filters,
                   activation='relu')
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
                       batch_norm=batch_norm)
      y = resnet_layer(y,
                       filters=filters,
                       activation=None,
                       batch_norm=batch_norm)
      if stack > 0 and res_block == 0:  # first layer but not first stack
        # linear projection residual shortcut connection to match changed dims
        x = resnet_layer(x,
                         filters=filters,
                         kernel_size=1,
                         strides=strides,
                         activation=None,
                         batch_norm=False)
      x = tf.keras.layers.add([x, y])
      x = tf.keras.layers.Activation('relu')(x)
    filters *= 2

  # v1 does not use BN after last shortcut connection-ReLU
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(num_classes, kernel_initializer='he_normal')(x)
  outputs = tf.keras.layers.Lambda(
      lambda inputs: ed.Categorical(logits=inputs))(x)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def get_metrics(model):
  """Get metrics for the model."""

  def negative_log_likelihood(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    return -y_pred.distribution.log_prob(y_true)

  def accuracy(y_true, y_pred):
    """Accuracy."""
    del y_pred  # unused arg
    y_true = tf.squeeze(y_true)
    return tf.equal(tf.argmax(input=model.output.distribution.logits, axis=1),
                    tf.cast(y_true, tf.int64))

  def log_marginal(y_true, y_pred):
    """Log-marginal likelihood."""
    del y_pred  # unused arg
    y_true = tf.squeeze(y_true)
    return model.output.distribution.log_prob(y_true)

  return negative_log_likelihood, accuracy, log_marginal


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  dataset_size = 40000
  dataset_train, ds_info = utils.load_dataset(tfds.Split.TRAIN, with_info=True)
  dataset_test = utils.load_dataset(tfds.Split.TEST)
  dataset_train = dataset_train.repeat().shuffle(10 * FLAGS.batch_size).batch(
      FLAGS.batch_size)
  validation_steps = 100
  dataset_test = dataset_test.take(FLAGS.batch_size * validation_steps).repeat(
      ).batch(FLAGS.batch_size)

  model = resnet_v1(input_shape=ds_info.features['image'].shape,
                    depth=20,
                    num_classes=ds_info.features['label'].num_classes,
                    batch_norm=FLAGS.batch_norm)
  negative_log_likelihood, accuracy, log_marginal = get_metrics(model)

  model.compile(tf.keras.optimizers.Adam(FLAGS.init_learning_rate),
                loss=negative_log_likelihood,
                metrics=[log_marginal, accuracy])
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
