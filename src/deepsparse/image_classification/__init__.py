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

# flake8: noqa
import importlib
import logging as _logging
import warnings
from collections import namedtuple


_LOGGER = _logging.getLogger(__name__)
_Dependency = namedtuple("_Dependency", ["name", "version", "necessary"])


def _auto_install_dependencies():
    dependencies = [
        _Dependency(name="torchvision", version=">=0.3.0,<=0.12.0", necessary=True),
    ]

    for dependency in dependencies:
        _check_and_install_dependency(dependency=dependency)


def _check_and_install_dependency(dependency: _Dependency):
    dependency_import_exception = _check_if_dependency_installed(
        dependency=dependency,
        raise_on_fail=False,
    )

    if not dependency_import_exception:
        return

    # attempt to install dependency
    import subprocess as _subprocess
    import sys as _sys

    install_name = f"{dependency.name}{dependency.version}"

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                install_name,
            ]
        )

        _check_if_dependency_installed(
            dependency=dependency,
            raise_on_fail=True,
        )

        _LOGGER.info(
            f"{dependency.name} dependency of deepsparse.image_classification "
            "sucessfully installed"
        )
    except Exception as dependency_exception:
        if dependency.necessary:
            raise ValueError(
                f"Unable to import or install {install_name}, a requirement of "
                f"deepsparse.image_classification. Failed with exception: "
                f"{dependency_exception}"
            )
        else:
            warnings.warn(
                message=f"Unable to import or install {install_name}",
                category=UserWarning,
            )


def _check_if_dependency_installed(dependency: _Dependency, raise_on_fail=False):
    try:
        _dep = importlib.import_module(dependency.name)
        return None
    except Exception as dependency_import_error:
        if raise_on_fail:
            raise dependency_import_error
        return dependency_import_error


_auto_install_dependencies()

from .constants import *
from .pipelines import *
from .schemas import *
