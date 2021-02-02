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

"""Stochastic output layers.

Stochastic output layers apply a linear layer to project from the input
dimension to the proper number of dimensions for parameterizing a distribution.
They also apply a set of constraints from the real-valued dimensions to the
domain for each parameter of the outputted random variable.

To avoid a bloated namespace where there is one stochastic output layer for
every available ed.RandomVariable, this module only implements stochastic
output layers that involve a lot of tensor manipulation (e.g., mixture
distributions and multi-parameter distributions with constraints).

For non-built-in stochastic output layers, we recommend you create your own:

```python
dataset_size = 10
batch_size = 2
num_classes = 5

numpy_features = np.random.normal(size=[dataset_size, 3]).astype('float32')
numpy_labels = np.random.randint(num_classes, size=dataset_size).astype('int32')
dataset = tf.data.Dataset.from_tensor_slices((numpy_features, numpy_labels))
dataset = dataset.repeat().batch(batch_size)

model = tf.python.keras.Sequential([
    tf.python.keras.layers.Dense(num_classes),
    tf.python.keras.layers.Lambda(lambda inputs: ed.Categorical(logits=inputs)),
])

model.compile(tf.python.keras.optimizers.Adam(0.1),
              loss=lambda y_true, y_pred: -y_pred.distribution.log_prob(y_true))
model.fit(dataset,
          steps_per_epoch=dataset_size // batch_size,
          epochs=10)
```
"""

from edward2.tensorflow import constraints
from edward2.tensorflow import generated_random_variables

import tensorflow as tf


class MixtureLogistic(tf.python.keras.layers.Layer):
  """Stochastic output layer, distributed as a mixture of logistics.

  Given an input tensor of shape [..., input_dim], the output layer returns
  an ed.Mixture of Logistic random variables of shape [...].
  """

  def __init__(self,
               num_components,
               logits_constraint=None,
               loc_constraint=None,
               scale_constraint='softplus',
               **kwargs):
    super(MixtureLogistic, self).__init__(**kwargs)
    self.num_components = num_components
    self.logits_constraint = constraints.get(logits_constraint)
    self.loc_constraint = constraints.get(loc_constraint)
    self.scale_constraint = constraints.get(scale_constraint)
    self.layer = tf.python.keras.layers.Dense(num_components * 3)

  def build(self, input_shape=None):
    self.layer.build(input_shape)
    self.built = True

  def call(self, inputs):
    net = self.layer(inputs)
    logits, loc, scale = tf.split(net, 3, axis=-1)
    if self.logits_constraint:
      logits = self.logits_constraint(logits)
    if self.loc_constraint:
      loc = self.loc_constraint(loc)
    if self.scale_constraint:
      scale = self.scale_constraint(scale)
    return generated_random_variables.MixtureSameFamily(
        mixture_distribution=generated_random_variables.Categorical(
            logits=logits).distribution,
        components_distribution=generated_random_variables.Logistic(
            loc=loc, scale=scale).distribution)

  def compute_output_shape(self, input_shape):
    return tf.TensorShape(input_shape)[:-1]

  def get_config(self):
    config = {'num_components': self.num_components}
    base_config = super(MixtureLogistic, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
