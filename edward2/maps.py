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

"""A better map."""

import concurrent.futures
from typing import Callable, Literal, overload, Sequence, TypeVar

from absl import logging
import grpc
import tenacity
import tqdm

T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: dict[int, U | V] | None = ...,
    max_retries: int | None = ...,
    max_backoff_seconds: int = 30,
    max_workers: int | None = ...,
    raise_error: Literal[False] = ...,
    retry_exception_types: list[type[Exception]] | None = ...,
    show_progress: bool = ...,
) -> list[U | V]:
  ...


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: dict[int, U | V] | None = ...,
    max_retries: int | None = ...,
    max_backoff_seconds: int = 30,
    max_workers: int | None = ...,
    raise_error: Literal[True] = ...,
    retry_exception_types: list[type[Exception]] | None = ...,
    show_progress: bool = ...,
) -> list[U]:
  ...


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: dict[int, U | V] | None = ...,
    max_retries: int | None = ...,
    max_backoff_seconds: int = 30,
    max_workers: int | None = ...,
    raise_error: bool = ...,
    retry_exception_types: list[type[Exception]] | None = ...,
    show_progress: bool = ...,
) -> list[U | V]:
  ...


@overload
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = ...,
    index_to_output: dict[int, U | V] | None = ...,
    max_retries: int | None = ...,
    max_backoff_seconds: int = 30,
    max_workers: int | None = ...,
    raise_error: bool = ...,
    progress_desc: str = ...,
    show_progress: bool = ...,
) -> list[U | V]:
  ...


# TODO(trandustin): Support nested structure inputs like jax.tree.map.
def robust_map(
    fn: Callable[[T], U],
    inputs: Sequence[T],
    error_output: V = None,
    index_to_output: dict[int, U | V] | None = None,
    max_retries: int | None = None,
    max_backoff_seconds: int = 30,
    max_workers: int | None = None,
    raise_error: bool = False,
    retry_exception_types: list[type[Exception]] | None = None,
    progress_desc: str = 'robust_map',
    show_progress: bool = True,
) -> list[U | V]:
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
    max_retries: The maximum number of times to retry each input. If None, then
      there is no limit. If limit, the output is set to `error_output` or an
      error is raised if `raise_error` is set to True.
    max_backoff_seconds: The maximum number of seconds to wait between retries
      when applying exponential backoff.
    max_workers: An optional maximum number of threadpool workers. If None, a
      default number will be used, which as of Python 3.8 is `min(32,
      os.cpu_count() + 4)`.
    raise_error: Whether to raise an error if an input exceeds `max_retries`.
      Will override any setting of `error_output`.
    retry_exception_types: Exception types to retry on. Defaults to retrying
      only on grpc's RPC exceptions.
    progress_desc: A string to display in the progress bar.
    show_progress: Whether to show the progress bar.

  Returns:
    A list of items each of type U. They are the outputs of `fn` applied to
    the elements of `inputs`.
  """
  if retry_exception_types is None:
    retry_exception_types = []
  retry_exception_types = retry_exception_types + [
      grpc.RpcError,
  ]
  retry_exception_types = list(set(retry_exception_types))
  retry = tenacity.retry_if_exception_type(retry_exception_types[0])
  for retry_exception_type in retry_exception_types[1:]:
    retry = retry | tenacity.retry_if_exception_type(retry_exception_type)
  if max_retries is None:
    fn_with_backoff = tenacity.retry(
        retry=retry,
        wait=tenacity.wait_random_exponential(min=1, max=max_backoff_seconds),
        before_sleep=tenacity.before_sleep_log(
            logging.get_absl_logger(), logging.WARNING
        ),
    )(fn)
  else:
    fn_with_backoff = tenacity.retry(
        retry=retry,
        wait=tenacity.wait_random_exponential(min=1, max=max_backoff_seconds),
        stop=tenacity.stop_after_attempt(max_retries + 1),
        before_sleep=tenacity.before_sleep_log(
            logging.get_absl_logger(), logging.WARNING
        ),
    )(fn)
  if index_to_output is None:
    index_to_output = {}
  num_existing = len(index_to_output)
  num_inputs = len(inputs)
  logging.info('Found %s/%s existing examples.', num_existing, num_inputs)
  if show_progress:
    progress_bar = tqdm.tqdm(
        total=num_inputs - num_existing, desc=progress_desc
    )
  else:
    progress_bar = None
  indices = [i for i in range(num_inputs) if i not in index_to_output.keys()]
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_workers
  ) as executor:
    future_to_index = {
        executor.submit(fn_with_backoff, inputs[i]): i for i in indices
    }
    for future in concurrent.futures.as_completed(future_to_index):
      index = future_to_index[future]
      try:
        output = future.result()
        index_to_output[index] = output
        if progress_bar:
          progress_bar.update(1)
      except tenacity.RetryError as e:
        if raise_error:
          logging.exception('Item %s exceeded max retries.', index)
          raise e
        else:
          logging.warning(
              'Item %s exceeded max retries. Output is set to %s. '
              'Exception: %s.',
              index,
              error_output,
              e,
          )
          index_to_output[index] = error_output
          if progress_bar:
            progress_bar.update(1)
  outputs = [index_to_output[i] for i in range(num_inputs)]
  return outputs
