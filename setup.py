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
import sys
from distutils import log
from fnmatch import fnmatch
from typing import Dict, List, Tuple

from setuptools import find_packages, setup
from setuptools.command.install import install


# default variables to be overwritten by the version.py file
is_release = None
version = "unknown"
version_major = version
version_minor = version
version_bug = version

# load and overwrite version and release info from deepsparse package
version_path = os.path.join("src", "deepsparse", "generated_version.py")
if not os.path.exists(version_path):
    version_path = os.path.join("src", "deepsparse", "version.py")
exec(open(version_path).read())
print(f"loaded version {version} from {version_path}")
version_base = f"{version_major}.{version_minor}.0"

_PACKAGE_NAME = "deepsparse" if is_release else "deepsparse-nightly"

# File regexes for binaries to include in package_data
binary_regexes = ["*/*.so", "*/*.so.*", "*.bin", "*/*.bin"]

_deps = ["numpy>=1.16.3", "onnx>=1.5.0,<=1.10.1", "requests>=2.0.0", "tqdm>=4.0.0"]
_nm_deps = [f"{'sparsezoo' if is_release else 'sparsezoo-nightly'}~={version_base}"]
_dev_deps = [
    "beautifulsoup4==4.9.3",
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "m2r2~=0.2.7",
    "myst-parser~=0.14.0",
    "rinohtype>=0.4.2",
    "sphinx>=3.4.0",
    "sphinx-copybutton>=0.3.0",
    "sphinx-markdown-tables>=0.0.15",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "sphinx-multiversion==0.2.4",
    "sphinx-rtd-theme",
    "onnxruntime>=1.4.0,<1.9.0",
    "flask>=1.0.0",
    "flask-cors>=3.0.0",
]
_transformers_deps = ["transformers~=4.8"]


class OverrideInstall(install):
    """
    Install class to run checks for supported systems before install
    and correcting binary file permissions after install.
    """

    def run(self):
        self._check_supported_system()
        self._check_supported_python_version()
        super().run()
        self._fix_file_modes()

    def _check_supported_system(self):
        if sys.platform.startswith("linux"):
            # linux is supported, allow install to go through
            return

        if sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
            # windows is not supported, raise error on install
            raise OSError(
                "Native Windows is currently unsupported for DeepSparse. "
                "Please run on a Linux system or within a Linux container on Windows. "
                "More info can be found in our docs here: "
                "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
            )

        if sys.platform.startswith("darwin"):
            # mac is not supported, raise error on install
            raise OSError(
                "Native Mac is currently unsupported for DeepSparse. "
                "Please run on a Linux system or within a Linux container on Mac. "
                "More info can be found in our docs here: "
                "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
            )

        # unknown system, raise error on install
        raise OSError(
            f"Unknown OS given of {sys.platform}, "
            "it is unsupported for DeepSparse. "
            "Please run on a Linux system. "
            "More info can be found in our docs here: "
            "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
        )

    def _check_supported_python_version(self):
        supported_major = 3
        supported_minor = [6, 7, 8, 9]

        if (
            sys.version_info[0] != supported_major
            or sys.version_info[1] not in supported_minor
        ):
            raise EnvironmentError(
                f"Python {supported_major}.{supported_minor} "
                f"is currently only supported for DeepSparse; found {sys.version}. "
                "Please run on a system with the proper Python version installed. "
                "More info can be found in our docs here: "
                "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
            )

    def _fix_file_modes(self):
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
    return _nm_deps + _deps


def _setup_extras() -> Dict:
    return {
        "dev": _dev_deps,
        "transformers": _transformers_deps,
    }


def _setup_entry_points() -> Dict:
    return {}


def _setup_long_description() -> Tuple[str, str]:
    return open("README.md", "r", encoding="utf-8").read(), "text/markdown"


setup(
    name=_PACKAGE_NAME,
    version=version,
    author="Neuralmagic, Inc.",
    author_email="support@neuralmagic.com",
    description=(
        "Neural network inference engine that delivers GPU-class performance "
        "for sparsified models on CPUs"
    ),
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
