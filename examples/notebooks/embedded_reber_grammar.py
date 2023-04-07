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

"""Embedded Reber grammar."""
import random

chars = 'BTPXVSTE'

# dictionary with states as keys, each value entry is the corresponding letter
# and the following state
reber_grammar = {
    0: [(1, 'B')],
    1: [(2, 'T'), (3, 'P')],
    2: [(2, 'S'), (5, 'X')],
    3: [(3, 'P'), (4, 'V')],
    4: [(5, 'P'), (6, 'V')],
    5: [(3, 'X'), (6, 'S')],
    6: [(-1, 'E')]
}

embedded_rg = {
    0: [(1, 'B')],
    1: [(2, 'T'), (3, 'P')],
    2: [(4, reber_grammar)],
    3: [(5, reber_grammar)],
    4: [(6, 'T')],
    5: [(6, 'P')],
    6: [(-1, 'E')]
}


def encode_string(string):
  encoding = {c: i for i, c in enumerate(chars)}
  encoded = []
  for c in string:
    zeros = [0 for i in range(len(chars))]
    zeros[encoding[c]] = 1
    encoded.append(zeros)
  return encoded


def generate_valid_string(grammar):
  """Generates a valid string."""
  state = 0
  string = ''
  while state >= 0:
    i = random.sample(list(range(len(grammar[state]))), 1)[0]
    next_char = grammar[state][i][1]

    if isinstance(next_char, dict):
      next_char = generate_valid_string(next_char)
    string += next_char
    state = grammar[state][i][0]
  return string


def generate_invalid_string(grammar):
  """Generates an invalid string."""
  string = generate_valid_string(grammar)
  set1 = chars[:4]
  set2 = chars[4:]
  strid = random.randrange(0, len(string))

  if string[strid] in set1:
    string = string[:strid]+set2[random.randrange(0, 4)]+string[strid+1:]
  else:
    string = string[:strid]+set1[random.randrange(0, 4)]+string[strid+1:]

  return string
