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

from utils.artifacts import (
    check_wand_binaries_exist,
    download_wand_binaries,
    get_release_and_version,
)


# Load version and release info from deepsparse package
package_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "src", "deepsparse"
)
(
    is_release,
    is_enterprise,
    version,
    version_major,
    version_minor,
    version_bug,
) = get_release_and_version(package_path)
version_base = f"{version_major}.{version_minor}.0"

_PACKAGE_NAME = (
    "deepsparse-ent"
    if is_enterprise
    else "deepsparse"
    if is_release
    else "deepsparse-nightly"
)

if is_enterprise:
    # do not include the LICENSE-NEURALMAGIC file
    # in the deepsparse-ent installation folder
    license_nm_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "LICENSE-NEURALMAGIC"
    )
    os.remove(license_nm_path)

# File regexes for binaries to include in package_data
binary_regexes = ["*/*.so", "*/*.so.*", "*.bin", "*/*.bin"]


def _parse_requirements_file(file_path):
    with open(file_path, "r") as requirements_file:
        lines = requirements_file.read().splitlines()

    return [line for line in lines if len(line) > 0 and line[0] != "#"]


_deps = [
    "numpy>=1.16.3",
    "onnx>=1.5.0,<=1.12.0",
    "pydantic>=1.8.2",
    "requests>=2.0.0",
    "tqdm>=4.0.0",
    "protobuf>=3.12.2,<=3.20.1",
    "click~=8.0.0",
]
_nm_deps = [f"{'sparsezoo' if is_release else 'sparsezoo-nightly'}~={version_base}"]
_dev_deps = [
    "beautifulsoup4>=4.9.3",
    "black>=20.8b1",
    "flake8>=3.8.3",
    "isort>=5.7.0",
    "m2r2~=0.2.7",
    "mistune==0.8.4",
    "myst-parser~=0.14.0",
    "ndjson>=0.3.1",
    "rinohtype>=0.4.2",
    "sphinx>=3.4.0",
    "sphinx-copybutton>=0.3.0",
    "sphinx-markdown-tables>=0.0.15",
    "wheel>=0.36.2",
    "pytest>=6.0.0",
    "sphinx-multiversion==0.2.4",
    "sphinx-rtd-theme",
    "onnxruntime>=1.7.0",
    "flask>=1.0.0",
    "flask-cors>=3.0.0",
    "Pillow>=8.3.2",
]
_server_deps = [
    "uvicorn>=0.15.0",
    "fastapi>=0.70.0",
    "pydantic>=1.8.2",
    "requests>=2.26.0",
    "python-multipart>=0.0.5",
    "prometheus-client>=0.14.1",
]
_onnxruntime_deps = [
    "onnxruntime>=1.7.0",
]
_yolo_integration_deps = [
    "torchvision>=0.3.0,<=0.12.0",
    "opencv-python",
]
# haystack dependencies are installed from a requirements file to avoid
# conflicting versions with NM's deepsparse/transformers
_haystack_requirements_file_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "src",
    "deepsparse",
    "transformers",
    "haystack",
    "haystack_reqs.txt",
)
_haystack_integration_deps = _parse_requirements_file(_haystack_requirements_file_path)


def _check_supported_system():
    if sys.platform.startswith("linux"):
        # linux is supported, allow install to go through
        return

    if sys.platform.startswith("win32") or sys.platform.startswith("cygwin"):
        # windows is not supported, raise error on install
        raise OSError(
            "Native Windows is currently unsupported for the DeepSparse Engine. "
            "Please run on a Linux system or within a Linux container on Windows. "
            "More info can be found in our docs here: "
            "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
        )

    if sys.platform.startswith("darwin"):
        # mac is not supported, raise error on install
        raise OSError(
            "Native Mac is currently unsupported for the DeepSparse Engine. "
            "Please run on a Linux system or within a Linux container on Mac. "
            "More info can be found in our docs here: "
            "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
        )

    # unknown system, raise error on install
    raise OSError(
        f"Unknown OS given of {sys.platform}; "
        "it is unsupported for the DeepSparse Engine. "
        "Please run on a Linux system. "
        "More info can be found in our docs here: "
        "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
    )


def _check_supported_python_version():
    supported_major = 3
    supported_minor = [7, 8, 9, 10]

    if (
        sys.version_info[0] != supported_major
        or sys.version_info[1] not in supported_minor
    ):
        raise EnvironmentError(
            f"Python {supported_major}.{supported_minor} "
            f"is only supported for the DeepSparse Engine; found {sys.version}. "
            "Please run on a system with the proper Python version installed. "
            "More info can be found in our docs here: "
            "https://docs.neuralmagic.com/deepsparse/source/hardware.html"
        )


# Ensure system and python environment is compatible
_check_supported_system()
_check_supported_python_version()

# Download WAND binaries if needed
if not check_wand_binaries_exist(package_path):
    download_wand_binaries(
        package_path, f"{version_major}.{version_minor}.{version_bug}", is_release
    )


class OverrideInstall(install):
    """
    Install class to run checks for correcting binary file permissions after install.
    """

    def run(self):
        super().run()
        self._fix_file_modes()

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
        "server": _server_deps,
        "onnxruntime": _onnxruntime_deps,
        "yolo": _yolo_integration_deps,
        "haystack": _haystack_integration_deps,
    }


def _setup_entry_points() -> Dict:
    data_api_entrypoint = "deepsparse.transformers.pipelines_cli:cli"
    eval_downstream = "deepsparse.transformers.eval_downstream:main"
    ic_eval = "deepsparse.image_classification.validation_script:main"

    return {
        "console_scripts": [
            f"deepsparse.transformers.run_inference={data_api_entrypoint}",
            f"deepsparse.transformers.eval_downstream={eval_downstream}",
            "deepsparse.analyze=deepsparse.analyze:main",
            "deepsparse.check_hardware=deepsparse.cpu:print_hardware_capability",
            "deepsparse.benchmark=deepsparse.benchmark.benchmark_model:main",
            "deepsparse.server=deepsparse.server.cli:main",
            "deepsparse.object_detection.annotate=deepsparse.yolo.annotate:main",
            "deepsparse.image_classification.annotate=deepsparse.image_classification.annotate:main",  # noqa E501
            "deepsparse.instance_segmentation.annotate=deepsparse.yolact.annotate:main",
            f"deepsparse.image_classification.eval={ic_eval}",
            "deepsparse.license=deepsparse.license:main",
            "deepsparse.validate_license=deepsparse.license:validate_license_cli",
        ]
    }


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
    python_requires=">=3.7, <3.11",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    cmdclass={"install": OverrideInstall},
)
