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

# Lint as: python2, python3
r"""Quantized ring of Gaussians with a discrete flow.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags
import edward2 as ed
from matplotlib import cm
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Number of elements in simulated data set.")
flags.DEFINE_integer("num_flows",
                     default=1,
                     help="Number of flows. 0 flows corresponds to modeling "
                          "with just the base distribution.")
flags.DEFINE_integer("hidden_size",
                     default=32,
                     help="Number of hidden units in LSTM or Transformer nets.")
flags.DEFINE_integer("max_steps",
                     default=500,
                     help="Number of steps to run optimizer.")
flags.DEFINE_integer("print_steps",
                     default=None,
                     help="Number of steps to print progress; default is "
                          "max_steps // 10.")
flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Fixed step-size in optimizer.")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "toy"),
    help="Directory to put the model's fit.")
flags.DEFINE_string("master",
                    default="",
                    help="BNS name of the TensorFlow master to use.")
FLAGS = flags.FLAGS


class EmbeddingNetwork(tf.keras.Model):
  """Autoregressive network which uniquely embeds each combination."""

  def __init__(self, output_size=None):
    """Initializes Embedding network.

    Args:
      output_size: Embedding output dimension. When `None`, `output_size`
        defaults to `vocab_size`, which are used for loc/scale modular networks.
        Sinkhorn networks require `output_size` to be `vocab_size ** 2`.
    """
    super(EmbeddingNetwork, self).__init__()
    self.output_size = output_size

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    data_size = input_shape[-2]
    num_classes = input_shape[-1]
    if isinstance(data_size, tf.Dimension):
      data_size = data_size.value
    if isinstance(num_classes, tf.Dimension):
      num_classes = num_classes.value
    if self.output_size is None:
      self.output_size = num_classes
    self.embeddings = []
    for dim in range(1, data_size):
      # Make each possible history unique by converting to a base 10 integer.
      embedding_layer = tf.keras.layers.Embedding(
          num_classes ** dim,
          self.output_size)
      self.embeddings.append(embedding_layer)

  def call(self, inputs, initial_state=None):
    """Returns Tensor of shape [..., data_size, output_size].

    Args:
      inputs: Tensor of shape [..., data_size, vocab_size].
      initial_state: `Tensor` of initial states corresponding to encoder output.
    """
    num_classes = inputs.shape[-1]
    sparse_inputs = tf.argmax(inputs, axis=-1, output_type=tf.int32)
    location_logits = [tf.zeros([tf.shape(sparse_inputs)[0], self.output_size])]
    for dim, embedding_layer in enumerate(self.embeddings, 1):
      powers = tf.pow(num_classes, tf.range(0, dim))
      embedding_indices = tf.reduce_sum(  # (batch_size,)
          sparse_inputs[:, :dim] * powers[tf.newaxis, :], axis=1)
      location_logits.append(embedding_layer(embedding_indices))

    location_logits = tf.stack(location_logits, axis=1)
    return location_logits


def sample_quantized_gaussian_mixture(batch_size):
  """Samples data from a 2D quantized mixture of Gaussians.

  This is a quantized version of the mixture of Gaussians experiment from the
  Unrolled GANS paper (Metz et al., 2017).

  Args:
    batch_size: The total number of observations.

  Returns:
    Tensor with shape `[batch_size, 2]`, where each entry is in
      `{0, 1, ..., max_quantized_value - 1}`, a rounded sample from a mixture
      of Gaussians.
  """
  clusters = np.array([[2., 0.], [np.sqrt(2), np.sqrt(2)],
                       [0., 2.], [-np.sqrt(2), np.sqrt(2)],
                       [-2., 0.], [-np.sqrt(2), -np.sqrt(2)],
                       [0., -2.], [np.sqrt(2), -np.sqrt(2)]])
  assignments = tfp.distributions.OneHotCategorical(
      logits=tf.zeros(8), dtype=tf.float32).sample(batch_size)
  means = tf.matmul(assignments, tf.cast(clusters, tf.float32))

  samples = tfp.distributions.Normal(loc=means, scale=0.1).sample()
  clipped_samples = tf.clip_by_value(samples, -2.25, 2.25)
  quantized_samples = tf.cast(tf.round(clipped_samples * 20) + 45, tf.int32)
  return quantized_samples


def main(argv):
  del argv  # unused
  tf.disable_v2_behavior()
  if FLAGS.print_steps is None:
    FLAGS.print_steps = FLAGS.max_steps // 10
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  data = sample_quantized_gaussian_mixture(FLAGS.batch_size)
  num_classes = 91
  data_size = 2

  tf.Session.reset(FLAGS.master)
  sess = tf.Session(FLAGS.master)

  one_hot_data = tf.one_hot(data, depth=num_classes, dtype=tf.float32)
  flows = []
  total_variance = tf.constant(0.)
  variances = tf.constant(0.)
  for flow_num in range(FLAGS.num_flows):
    output_size = num_classes
    flow_network = EmbeddingNetwork(output_size=output_size)

    temperature = 0.1
    flow = ed.layers.DiscreteAutoregressiveFlow(flow_network, temperature)
    # Use right-to-left ordering for every other flow. We do so by reversing
    # the flow's inputs and outputs. A more efficient implentation would
    # change the ordering within the network.
    one_hot_data = tf.reverse(one_hot_data, axis=[-2])
    flow_output = flow.reverse(one_hot_data)

    # Monitor variances of location output.
    locs = tf.cast(tf.argmax(flow_output, -1) - tf.argmax(one_hot_data, -1),
                   tf.int32)
    locs = tf.cast(tf.mod(locs, num_classes), tf.float32)
    _, variances = tf.nn.moments(locs, axes=[0])
    variances = tf.reshape(variances, [-1])
    tf.summary.histogram("variances/flow_network_{}".format(flow_num),
                         variances)
    if data_size <= 10:
      for data_dim in range(data_size):
        tf.summary.scalar(
            "variances/flow_network_{}_dim_{}".format(flow_num, data_dim),
            variances[data_dim])
    total_variance = tf.reduce_sum(variances)

    one_hot_data = flow_output
    if flow_num % 2 == 1:
      one_hot_data = tf.reverse(one_hot_data, axis=[-2])
    flows.append(flow)

  base_logits = tf.get_variable("base_logits",
                                [data_size, num_classes],
                                dtype=tf.float32)
  logits = tf.broadcast_to(base_logits, one_hot_data.shape)
  data_log_prob = -tf.reduce_sum(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_data,
                                                 logits=logits),
      axis=-1)

  loss = -tf.reduce_mean(data_log_prob)

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
  grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
  train_op = optimizer.apply_gradients(grads_and_vars)

  tf.summary.scalar("loss", loss)
  for grad, var in grads_and_vars:
    var_name = var.name.replace(":", "/")  # colons are invalid characters
    tf.summary.histogram("gradient/{}".format(var_name), grad)
    tf.summary.scalar("gradient_norm/{}".format(var_name), tf.norm(grad))
    if len(var.shape) == 0:  # pylint: disable=g-explicit-length-test
      tf.summary.scalar("parameter/{}".format(var_name), var)
    elif len(var.shape) == 1 and var.shape[0] == 1:
      tf.summary.scalar("parameter/{}".format(var_name), var[0])
    else:
      tf.summary.histogram("parameter/{}".format(var_name), var)

  print("Number of sets of parameters: {}".format(
      len(tf.trainable_variables())))
  print("Number of parameters: {}".format(
      np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])))
  for v in tf.trainable_variables():
    print(v)

  summary = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)

  init = tf.global_variables_initializer()
  sess.run(init)

  for step in range(FLAGS.max_steps):
    _, loss_value, total_variance_value = sess.run(
        [train_op, loss, total_variance])
    if step % FLAGS.print_steps == 0 or step == FLAGS.max_steps - 1:
      print("Step: {:>3d} "
            "NLL: {:.3f} "
            "Total Variance: {:.3f} ".format(step,
                                             loss_value,
                                             total_variance_value))
      print("Flow Variances:", sess.run(variances))
      summary_str = sess.run(summary)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

  if FLAGS.num_flows > 0:
    lookup_table = sess.run(
        tf.argmax(flow_network.embeddings[0].weights[0], -1))
    print("First-Dimension Lookup Table from Last Flow:", lookup_table)
    base_probs_val = sess.run(tf.nn.softmax(base_logits))
    if data_size < 10 and num_classes < 20:
      for dim in range(data_size):
        print("Estimated base probs {:>2d}:".format(dim) +
              (num_classes * " {:.3f}").format(*base_probs_val[dim]))

  # Generate 10 samples.
  data_distribution = tfp.distributions.OneHotCategorical(logits=base_logits,
                                                          dtype=tf.float32)
  samples = data_distribution.sample(FLAGS.batch_size)

  # Add flows on top of base samples.
  for flow_num in reversed(list(range(FLAGS.num_flows))):
    flow = flows[flow_num]
    if flow_num % 2 == 1:
      samples = tf.reverse(samples, axis=[-2])
    samples = flow(samples)
    if flow_num % 2 == 1:
      samples = tf.reverse(samples, axis=[-2])
    print("Flow {} Variances:".format(flow_num), sess.run(variances))

  data_samples = tf.argmax(samples, axis=-1)
  data_val, samples_val = sess.run([data, data_samples])
  print("Data", data_val[:7])
  print("Samples", samples_val[:7])
  figsize = (12, 6)
  fig = figure.Figure(figsize=figsize)
  canvas = backend_agg.FigureCanvasAgg(fig)
  ax1 = fig.add_subplot(1, 2, 1)
  ax2 = fig.add_subplot(1, 2, 2)
  data_val = sess.run(data)
  data_prob_table = np.histogramdd(data_val, bins=num_classes)
  ax1.imshow(data_prob_table[0]/np.sum(data_prob_table[0]),
             cmap=cm.get_cmap("Blues", 6),
             origin="lower",
             extent=[0, num_classes, 0, num_classes],
             interpolation="nearest")
  ax1.set_title("Data Distribution")
  ax2.set_title("{} ({} {} Flows)".format(
      "factorized".replace("_", " ").title(),
      "embedding".title(),
      FLAGS.num_flows))
  learned_prob_table = np.histogramdd(samples_val, bins=num_classes)
  ax2.imshow(learned_prob_table[0]/np.sum(learned_prob_table[0]),
             cmap=cm.get_cmap("Blues", 6),
             origin="lower",
             extent=[0, num_classes, 0, num_classes],
             interpolation="nearest")

  if FLAGS.num_flows > 0:
    fname = "{}_embedding_factorized.png".format(FLAGS.num_flows)
  else:
    fname = "factorized.png"
  fname = os.path.join(FLAGS.model_dir, fname)
  canvas.print_figure(fname, format="png")
  print("Saved {}".format(fname))

if __name__ == "__main__":
  tf.app.run()
