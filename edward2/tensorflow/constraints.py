# coding=utf-8
# Copyright 2022 The Edward2 Authors.
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

"""Constraints.

One subtlety is how Bayesian Layers uses `tf.keras.constraints`. Typically,
Keras constraints are used with projected gradient descent, where one performs
unconstrained optimization and then applies a projection (the constraint) after
each gradient update. To stay in line with probabilistic literature, trainable
initializers, such as variational distributions for the weight initializer,
apply constraints on the `tf.Variables` themselves (i.e., a constrained
parameterization) and do not apply projections during optimization.
"""

import tensorflow as tf


class Exp(tf.keras.constraints.Constraint):
  """Exp constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.exp(w) + self.epsilon

  def get_config(self):
    return {'epsilon': self.epsilon}


class Positive(tf.keras.constraints.Constraint):
  """Positive constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.maximum(w, self.epsilon)

  def get_config(self):
    return {'epsilon': self.epsilon}


class Softplus(tf.keras.constraints.Constraint):
  """Softplus constraint."""

  def __init__(self, epsilon=tf.keras.backend.epsilon()):
    self.epsilon = epsilon

  def __call__(self, w):
    return tf.nn.softplus(w) + self.epsilon

  def get_config(self):
    return {'epsilon': self.epsilon}


# Compatibility aliases, following tf.keras

# pylint: disable=invalid-name
exp = Exp
positive = Positive
softplus = Softplus
# pylint: enable=invalid-name

# Utility functions, following tf.keras


def serialize(initializer):
  return tf.keras.utils.legacy.serialize_keras_object(initializer)


def deserialize(config, custom_objects=None):
  return tf.keras.utils.legacy.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraints',
  )


def get(identifier, value=None):
  """Getter for loading from strings; falls back to Keras as needed."""
  if value is None:
    value = identifier
  if identifier is None:
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      pass
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      pass
  elif callable(identifier):
    return identifier
  return tf.keras.constraints.get(value)
