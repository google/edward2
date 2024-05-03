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

"""Tracing mechanism for controlling the execution of programs."""

import contextlib
import functools
import threading


class _TracerStack(threading.local):
  """A thread-local stack of tracers."""

  def __init__(self):
    super(_TracerStack, self).__init__()
    self.stack = [lambda f, *args, **kwargs: f(*args, **kwargs)]


_tracer_stack = _TracerStack()


@contextlib.contextmanager
def trace(tracer):
  """Python context manager for tracing.

  Upon entry, a trace context manager pushes an tracer onto a
  thread-local stack. Upon exiting, it pops the tracer from the stack.

  Args:
    tracer: Function which takes a callable `f` and inputs `*args`, `**kwargs`.

  Yields:
    None.

  #### Examples

  Tracing controls the execution of Edward programs. Below we illustrate
  how to set the value of a specific random variable within a program.

  ```python
  import edward2 as ed

  def model():
    return ed.Poisson(rate=1.5, name="y")

  def tracer(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 42
    return ed.traceable(f)(*args, **kwargs)

  with ed.trace(tracer):
    y = model()

  assert y == 42
  ```

  Wrapping `f` as `traceable` allows tracers down the stack to
  additionally modify this operation. Since the operation `f()` is not wrapped
  by default, we could have called it directly. Refer also to the example in
  `get_next_tracer()` for more details on nested tracers.
  """
  try:
    _tracer_stack.stack.append(tracer)
    yield
  finally:
    _tracer_stack.stack.pop()


@contextlib.contextmanager
def get_next_tracer():
  """Yields the top-most tracer on the thread-local trace stack.

  Operations may be traced by multiple nested tracers. Once reached,
  an operation can be forwarded through nested tracers until resolved.
  To allow for nesting, implement tracers by re-wrapping their first
  argument (`f`) as an `traceable`. To avoid nesting, manipulate the
  computation without using `traceable`.

  This function allows for nesting by manipulating the thread-local tracer
  stack, so that operations are traced in the order of tracer nesting.

  #### Examples

  ```python
  import edward2 as ed

  def model():
    x = ed.Normal(loc=0., scale=1., name="x")
    y = ed.Normal(loc=x, scale=1., name="y")
    return x + y

  def double(f, *args, **kwargs):
    return 2. * ed.traceable(f)(*args, **kwargs)

  def set_y(f, *args, **kwargs):
    if kwargs.get("name") == "y":
      kwargs["value"] = 0.42
    return ed.traceable(f)(*args, **kwargs)

  with ed.trace(double):
    with ed.trace(set_y):
      z = model()
  ```

  This will firstly put `double` on the stack, and then `set_y`,
  resulting in the stack:
  (TOP) set_y -> double -> apply (BOTTOM)

  The execution of `model` is then (top lines are current stack state):
  1) (TOP) set_y -> double -> apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is traced by `set_y`, and as the name is not "y"
  the operation is simply forwarded to the next tracer on the stack.

  2) (TOP) double -> apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is traced by `double`, to produce
  `2*ed.Normal(0., 1., "x")`, with the operation being forwarded down the stack.

  3) (TOP) apply (BOTTOM);
  `ed.Normal(0., 1., "x")` is traced by `apply`, which simply calls the
  constructor.

  (At this point, the nested calls to `get_next_tracer()`, produced by
  forwarding operations, exit, and the current stack is again:
  (TOP) set_y -> double -> apply (BOTTOM))

  4) (TOP) set_y -> double -> apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is traced by `set_y`,
  the value of `y` is set to 0.42 and the operation is forwarded down the stack.

  5) (TOP) double -> apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is traced by `double`, to produce
  `2*ed.Normal(0., 1., "y")`, with the operation being forwarded down the stack.

  6) (TOP) apply (BOTTOM);
  `ed.Normal(0., 1., "y")` is traced by `apply`, which simply calls the
  constructor.

  The final values for `x` and `y` inside of `model()` are tensors where `x` is
  a random draw from Normal(0., 1.) doubled, and `y` is a constant 0.84, thus
  z = 2 * Normal(0., 1.) + 0.84.
  """
  tracer = _tracer_stack.stack.pop()
  try:
    yield tracer
  finally:
    _tracer_stack.stack.append(tracer)


def traceable(func):
  """Decorator that wraps `func` so that its execution is traced.

  The wrapper passes `func` to the tracer for the current thread.

  If there is no next tracer, we perform an "immediate" call to `func`.
  That is, `func` terminates without forwarding its execution to another
  tracer.

  Args:
    func: Function to wrap.

  Returns:
    The decorated function.
  """
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    with get_next_tracer() as tracer:
      return tracer(func, *args, **kwargs)

  return func_wrapped
