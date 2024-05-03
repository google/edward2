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

"""Tests maps."""

import edward2 as ed
from edward2 import maps
import grpc
import numpy as np
import tenacity
import tensorflow as tf


class MapsTest(tf.test.TestCase):

  def test_robust_map(self):
    """Tests the default call under the direct import."""
    x = [0, 1, 2]
    fn = lambda x: x + 1
    y = maps.robust_map(fn, x)
    self.assertEqual(y, [1, 2, 3])

  def test_robust_map_library_import(self):
    """Tests the default call under the library import."""
    x = [0, 1, 2]
    fn = lambda x: x + 1
    y = ed.robust_map(fn, x)
    self.assertEqual(y, [1, 2, 3])

  def test_robust_map_error_output(self):
    def fn(x):
      if x == 1:
        raise grpc.RpcError('Input value 1 takes too long to process.')
      else:
        return x + 1

    x = [0, 1, 2]
    y = maps.robust_map(
        fn,
        x,
        error_output=np.nan,
        max_retries=1,
    )
    self.assertEqual(y, [1, np.nan, 3])

  def test_robust_map_index_to_output(self):
    x = [1, 2, 3]
    fn = lambda x: x + 1
    index_to_output = {0: 2}
    y = maps.robust_map(
        fn,
        x,
        index_to_output=index_to_output,
    )
    self.assertEqual(y, [2, 3, 4])
    self.assertEqual(index_to_output, {0: 2, 1: 3, 2: 4})

  def test_robust_map_max_retries(self):
    def fn(x):
      if x == 1:
        raise grpc.RpcError('Input value 1 takes too long to process.')
      else:
        return x + 1

    x = [0, 1, 2]
    y = maps.robust_map(
        fn,
        x,
        max_retries=1,
    )
    self.assertEqual(y, [1, None, 3])

  def test_robust_map_raise_error(self):
    def fn(x):
      if x == 1:
        raise grpc.RpcError('Input value 1 is not supported.')
      else:
        return x + 1

    x = [0, 1, 2]
    with self.assertRaises(tenacity.RetryError):
      maps.robust_map(
          fn,
          x,
          max_retries=1,
          raise_error=True,
      )

  def test_robust_map_non_rpc_error(self):
    def fn(x):
      if x == 1:
        raise ValueError('Input value 1 is not supported.')
      else:
        return x + 1

    x = [0, 1, 2]
    with self.assertRaises(ValueError):
      maps.robust_map(fn, x)

  def test_robust_map_retry_exception_types(self):

    def make_fn():
      busy = True
      def fn(x):
        nonlocal busy
        if busy:
          busy = False
          raise RuntimeError("Sorry, can't process request right now.")
        else:
          busy = True
          return x + 1
      return fn

    fn = make_fn()
    x = [0, 1, 2]
    y = maps.robust_map(fn, x, retry_exception_types=[RuntimeError])
    self.assertEqual(y, [1, 2, 3])


if __name__ == '__main__':
  tf.test.main()
