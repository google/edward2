#!/bin/bash
# Releases a new Edward2 package for Python Package Index.

set -v  # print commands as they're executed
set -e  # fail and exit on any command erroring

GIT_COMMIT_ID=${1:-""}
[[ -z $GIT_COMMIT_ID ]] && echo "Must provide a commit" && exit 1

TMP_DIR=$(mktemp -d)
pushd $TMP_DIR

echo "Cloning google/edward2 and checking out commit $GIT_COMMIT_ID"
git clone https://github.com/google/edward2.git
cd edward2
git checkout $GIT_COMMIT_ID

pip install wheel twine pyopenssl

# Build the distribution
echo "Building distribution"
python setup.py sdist
python setup.py bdist_wheel --universal

# Publish to PyPI
read -p "Publish? (y/n) " -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
  echo "Publishing to PyPI"
  twine upload dist/*
else
  echo "Skipping upload"
  exit 1
fi

# Cleanup
rm -rf build/ dist/ edward2.egg-info/
popd
rm -rf $TMP_DIR
