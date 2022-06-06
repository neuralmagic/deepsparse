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

import logging as _logging


_LOGGER = _logging.getLogger(__name__)


def _check_torchvision_install(raise_on_fail=False):
    try:
        import torchvision as _torchvision

        return None
    except Exception as torchvision_import_exception:
        if raise_on_fail:
            raise torchvision_import_exception
        return torchvision_import_exception


def _check_click_install(raise_on_fail=False):
    try:
        import click as _click

        return None
    except Exception as click_import_exception:
        if raise_on_fail:
            raise click_import_exception
        return click_import_exception


def _check_install_deps():
    torchvision_import_exception = _check_torchvision_install
    if not torchvision_import_exception:
        return

    # attempt to install torchvision
    import subprocess as _subprocess
    import sys as _sys

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "torchvision>=0.3.0,<=0.10.1",
            ]
        )

        _check_torchvision_install(raise_on_fail=True)

        _LOGGER.info(
            "torchvision dependency of deepsparse.image_classification "
            "sucessfully installed"
        )
    except Exception as torchvision_exception:
        raise ValueError(
            "Unable to import or install torchvision, a requirement of "
            f"deepsparse.image_classification. Failed with exception: "
            f"{torchvision_exception}"
        )

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                "click<8.1",
            ]
        )

        _check_click_install(raise_on_fail=True)

        _LOGGER.info(
            "click dependency of deepsparse.image_classification "
            "sucessfully installed"
        )
    except Exception as click_exception:
        raise ValueError(
            "Unable to import or install click, a requirement of "
            f"deepsparse.image_classification. Failed with exception: "
            f"{click_exception}"
        )


_check_install_deps()


from .constants import *
from .pipelines import *
from .schemas import *
