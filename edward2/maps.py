# coding=utf-8
# Copyright 2023 The Edward2 Authors.
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

"""A better map."""

import concurrent.futures
from typing import Callable, Literal, Optional, Sequence, TypeVar, overload

from absl import logging
import tenacity

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: Optional[dict[int, U | V]] = ...,
    log_percent: Optional[int | float] = ...,
    max_retries: Optional[int] = ...,
    max_workers: Optional[int] = ...,
    raise_error: Literal[False] = ...,
) -> Sequence[U | V]:
  ...


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: Optional[dict[int, U | V]] = ...,
    log_percent: Optional[int | float] = ...,
    max_retries: Optional[int] = ...,
    max_workers: Optional[int] = ...,
    raise_error: Literal[True] = ...,
) -> Sequence[U]:
  ...


# TODO(trandustin): Support nested structure inputs like jax.tree_map.
# TODO(trandustin): Replace the infinite outer loop retry with tenacity's
# before_sleep feature.
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = None,
    index_to_output: Optional[dict[int, U | V]] = None,
    log_percent: Optional[int | float] = 5,
    max_retries: Optional[int] = None,
    max_workers: Optional[int] = None,
    raise_error: bool = False,
) -> Sequence[U | V]:
  """Maps a function to inputs using a threadpool.

  The map supports exception handling, retries with exponential backoff, and
  in-place updates in order to store intermediate progress.

  Args:
    fn: A function that takes in a type T and returns a type U.
    inputs: A list of items each of type T.
    error_output: Value to set as function's output if an input exceeds
      `max_retries`.
    index_to_output: Optional dictionary to be used to store intermediate
      results in-place.
    log_percent: At every `log_percent` percent of items, log the progress.
    max_retries: The maximum number of times to retry each input. If None, then
      there is no limit. If limit, the output is set to `error_output`.
    max_workers: An optional maximum number of threadpool workers. If None, a
      default number will be used, which as of Python 3.8 is `min(32,
      os.cpu_count() + 4)`.
    raise_error: Whether to raise an error if an input exceeds `max_retries`.
      Will override any setting of `error_output`.

  Returns:
    A list of items each of type U. They are the outputs of `fn` applied to
    the elements of `inputs`.
  """
  if index_to_output is None:
    index_to_output = {}
  if max_retries is None:
    # Apply exponential backoff with 3 retries. Retry infinitely in outer loop.
    fn_retries = 3
  else:
    fn_retries = max_retries
  fn_with_backoff = tenacity.retry(
      wait=tenacity.wait_random_exponential(min=1, max=60),
      stop=tenacity.stop_after_attempt(fn_retries),
  )(fn)
  num_inputs = len(inputs)
  log_steps = max(1, num_inputs * log_percent // 100)
  indices = [i for i in range(num_inputs) if i not in index_to_output.keys()]
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    while indices:
      future_to_index = {
          executor.submit(fn_with_backoff, inputs[i]): i for i in indices
      }
      indices = []  # Clear the list since the tasks have been submitted.
      for future in concurrent.futures.as_completed(future_to_index):
        index = future_to_index[future]
        try:
          output = future.result()
          index_to_output[index] = output
        except tenacity.RetryError as e:
          if max_retries is not None and raise_error:
            logging.exception('Item %s exceeded max retries.', index)
            raise e
          elif max_retries is not None:
            logging.warning(
                'Item %s exceeded max retries. Output is set to %s. '
                'Exception: %s.',
                index,
                error_output,
                e,
            )
            index_to_output[index] = error_output
          else:
            logging.info('Retrying item %s after exception: %s.', index, e)
            indices.append(index)
        processed_len = len(index_to_output)
        if processed_len % log_steps == 0 or processed_len == num_inputs:
          logging.info(
              'Completed %s/%s inputs, with %s left to retry.',
              processed_len,
              num_inputs,
              len(indices),
          )
  outputs = [index_to_output[i] for i in range(num_inputs)]
  return outputs