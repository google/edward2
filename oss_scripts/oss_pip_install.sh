#!/bin/bash
# Installs Edward2 with various dependencies for unit testing.

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

# Ensure that the base dependencies are sufficient for a full import.
pip install -q -e .
python -c "import edward2 as ed"

# Install backend and test dependencies.
if [[ "$TF_VERSION" == "tensorflow"  ]]
then
  pip install -q -e .[numpy,tensorflow,tests]
else
  pip install -q -e .[numpy,tf-nightly,tests]
fi
