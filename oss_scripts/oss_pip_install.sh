#!/bin/bash
# Installs Edward2 with various dependencies for unit testing.

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

: "${TF_VERSION:?}"

if [[ "$TF_VERSION" == "tf-nightly"  ]]
then
  pip install tf-nightly
  pip install tfp-nightly
  pip install -q -e .[numpy]
else
  pip install -q -e .[numpy,tensorflow]
fi

# Ensure that the base dependencies are sufficient for a full import.
python -c "import edward2 as ed"

# Install test dependencies.
pip install -q -e .[tests]
