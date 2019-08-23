#!/bin/bash
# Runs Python linter.

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

# Run linter.
rm -rf edward2.egg-info/
pylint --jobs=2 --rcfile=pylintrc *.py
pylint --jobs=2 --rcfile=pylintrc */
