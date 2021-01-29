# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from distutils import log
from fnmatch import fnmatch
from typing import Dict, List, Tuple

from setuptools import find_packages, setup
from setuptools.command.install import install


# File regexes for binaries to include in package_data
binary_regexes = ["*/*.so", "*/*.so.*", "*.bin", "*/*.bin"]


_deps = ["numpy>=1.16.3", "onnx>=1.5.0,<1.8.0", "requests>=2.0.0", "sparsezoo>=0.1.0"]

_dev_deps = [
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "rinohtype>=0.4.2",
    "sphinxcontrib-apidoc>=0.3.0",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "sphinx-rtd-theme",
]


class OverrideInstall(install):
    """
    This class adds a hook that runs after regular install that
    changes the permissions of all the binary files to 0755.
    """

    def run(self):
        install.run(self)
        mode = 0o755
        for filepath in self.get_outputs():
            if any(fnmatch(filepath, regex) for regex in binary_regexes):
                log.info("changing mode of %s to %s" % (filepath, oct(mode)))
                os.chmod(filepath, mode)


def _setup_package_dir() -> Dict:
    return {"": "src"}


def _setup_packages() -> List:
    return find_packages(
        "src", include=["deepsparse", "deepsparse.*"], exclude=["*.__pycache__.*"]
    )


def _setup_package_data() -> Dict:
    return {"deepsparse": binary_regexes}


def _setup_install_requires() -> List:
    return _deps


def _setup_extras() -> Dict:
    return {"dev": _dev_deps}


def _setup_entry_points() -> Dict:
    return {}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name="deepsparse",
    version="0.1.0",
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description="CPU runtime that delivers unprecedented performance for sparse models",
    long_description=_setup_long_description()[0],
    long_description_content_type=_setup_long_description()[1],
    keywords=(
        "inference, machine learning, x86, x86_64, avx2, avx512, neural network, "
        "sparse, inference engine, cpu, runtime, deepsparse, computer vision, "
        "object detection, sparsity"
    ),
    license="Neural Magic Engine License, Apache",
    url="https://github.com/neuralmagic/deepsparse",
    package_dir=_setup_package_dir(),
    include_package_data=True,
    package_data=_setup_package_data(),
    packages=_setup_packages(),
    install_requires=_setup_install_requires(),
    extras_require=_setup_extras(),
    entry_points=_setup_entry_points(),
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"install": OverrideInstall},
)
