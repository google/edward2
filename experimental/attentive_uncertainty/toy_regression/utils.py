# coding=utf-8
# Copyright 2024 The Edward2 Authors.
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

"""Defined utility functions for toy regression problem.
"""

import matplotlib.pyplot as plt


def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
  """Plots the predicted mean and variance and the context points.

  Args:
    target_x: An array of shape [B, num_targets, 1] that contains the
        x values of the target points.
    target_y: An array of shape [B, num_targets, 1] that contains the
        y values of the target points.
    context_x: An array of shape [B, num_contexts, 1] that contains
        the x values of the context points.
    context_y: An array of shape [B, num_contexts, 1] that contains
        the y values of the context points.
    pred_y: An array of shape [B, num_targets, 1] that contains the
        predicted means of the y values at the target points in target_x.
    std: An array of shape [B, num_targets, 1] that contains the
        predicted std dev of the y values at the target points in target_x.
  """
  # Plot everything
  plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
  plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
  plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
  plt.fill_between(
      target_x[0, :, 0],
      pred_y[0, :, 0] - std[0, :, 0],
      pred_y[0, :, 0] + std[0, :, 0],
      alpha=0.2,
      facecolor='#65c9f7',
      interpolate=True)

  # Make the plot pretty
  plt.yticks([-2, 0, 2], fontsize=16)
  plt.xticks([-2, 0, 2], fontsize=16)
  plt.ylim([-2, 2])
  plt.grid(False)
  plt.show()
