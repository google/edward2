"""Edward2.

Edward2 is a probabilistic programming language in Python. It extends the NumPy
or TensorFlow ecosystem so that one can declare models as probabilistic programs
and manipulate a model's computation for flexible training, latent variable
inference, and prediction.

See more details in the [`README.md`](https://github.com/google/edward2).
"""

import os
import sys

from setuptools import find_packages
from setuptools import setup

# To enable importing version.py directly, we add its path to sys.path.
version_path = os.path.join(os.path.dirname(__file__), 'edward2')
sys.path.append(version_path)
from version import __version__  # pylint: disable=g-import-not-at-top

setup(
    name='edward2',
    version='0.0.1',
    description='Edward2',
    author='Edward2 Team',
    author_email='trandustin@google.com',
    url='http://github.com/google/edward2',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'six',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.13.0',
                       'tensorflow-probability>=0.4.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.13.0',
                           'tensorflow-probability-gpu>=0.4.0'],
        'numpy': ['numpy>=1.7',
                  'scipy>=1.0.0'],
        'tests': [
            'absl-py',
            'pytest',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='probabilistic programming tensorflow machine learning',
)
