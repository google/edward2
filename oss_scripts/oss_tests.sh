#!/bin/bash
# Runs Edward2's unit tests.

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

# Run tests.
pytest
