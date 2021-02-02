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

"""Bayesian embedding layers."""

from edward2.tensorflow import constraints
from edward2.tensorflow import initializers
from edward2.tensorflow import regularizers
from edward2.tensorflow.layers import utils

import tensorflow as tf


@utils.add_weight
class EmbeddingReparameterization(tf.python.keras.layers.Embedding):
  """Bayesian embedding layer estimated via reparameterization.

  The layer computes a variational Bayesian approximation to the distribution
  over embedding layer functions,

  ```
  p(outputs | inputs) = int embedding(inputs; weights) p(weights) dweights.
  ```

  It does this with a stochastic forward pass, sampling from learnable
  distributions on the embedding weights. Gradients with respect to the
  distributions' learnable parameters backpropagate via reparameterization.
  Minimizing cross-entropy plus the layer's losses performs variational minimum
  description length, i.e., it minimizes an upper bound to the negative marginal
  likelihood.
  """

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='trainable_normal',
               embeddings_regularizer='normal_kl_divergence',
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               **kwargs):
    """Initializes the reparameterized Bayesian embeddings layer.

    Args:
      input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index +
      1.
      output_dim: int >= 0. Dimension of the dense embedding.
      embeddings_initializer: Initializer for the `embeddings` matrix.
      embeddings_regularizer: Regularizer function applied to the `embeddings`
        matrix.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      embeddings_constraint: Constraint function applied to the `embeddings`
        matrix.
      mask_zero: Whether or not the input value 0 is a special "padding" value
        that should be masked out.  This is useful when using recurrent layers
        which may take variable length input.  If this is `True` then all
        subsequent layers in the model need to support masking or an exception
        will be raised.  If mask_zero is set to True, as a consequence, index 0
        cannot be used in the vocabulary (input_dim should equal size of
        vocabulary + 1).
      input_length: Length of input sequences, when it is constant.  This
        argument is required if you are going to connect `Flatten` then `Dense`
        layers upstream (without it, the shape of the dense outputs cannot be
        computed).
      **kwargs: Additional keyword arguments to pass to the super class.
    """
    super(EmbeddingReparameterization, self).__init__(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer=initializers.get(embeddings_initializer),
        embeddings_regularizer=regularizers.get(embeddings_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        embeddings_constraint=constraints.get(embeddings_constraint),
        mask_zero=mask_zero,
        input_length=input_length,
        **kwargs)

  def call_weights(self):
    """Calls any weights if the initializer is itself a layer."""
    if isinstance(self.embeddings_initializer, tf.python.keras.layers.Layer):
      self.embeddings = self.embeddings_initializer(self.embeddings.shape,
                                                    self.dtype)

  def call(self, *args, **kwargs):
    """Computes the forward pass of this function."""
    self.call_weights()
    kwargs.pop('training', None)
    return super(EmbeddingReparameterization, self).call(*args, **kwargs)
