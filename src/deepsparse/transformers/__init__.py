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

"""
Helpers for running transformer based models with DeepSparse and integrating with
huggingface/transformers
"""

# flake8: noqa

import logging as _logging


try:
    import transformers as _transformers

    # triggers error if neuralmagic/transformers is not installed
    assert _transformers.NM_INTEGRATED
    _transformers_import_error = None
except Exception as _transformers_import_err:
    _transformers_import_error = _transformers_import_err


_LOGGER = _logging.getLogger(__name__)
_NM_TRANSFORMERS_TAR_TEMPLATE = (
    "https://github.com/neuralmagic/transformers/releases/download/"
    "{version}/transformers-4.23.1-py3-none-any.whl"
)
_NM_TRANSFORMERS_NIGHTLY = _NM_TRANSFORMERS_TAR_TEMPLATE.format(version="nightly")


def _install_transformers_and_deps():

    import subprocess as _subprocess
    import sys as _sys

    import deepsparse as _deepsparse

    nm_transformers_release = (
        "nightly"
        if not _deepsparse.is_release
        else f"v{_deepsparse.version.version_major_minor}"
    )
    transformers_requirement = _NM_TRANSFORMERS_TAR_TEMPLATE.format(
        version=nm_transformers_release
    )

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                transformers_requirement,
                "datasets<=1.18.4",
                "sklearn",
                "seqeval",
            ]
        )

        import transformers as _transformers

        _LOGGER.info("deepsparse-transformers and dependencies successfully installed")
    except Exception:
        raise ValueError(
            "Unable to install and import deepsparse-transformers dependencies check "
            "that transformers is installed, if not, install via "
            f"`pip install {_NM_TRANSFORMERS_NIGHTLY}`"
        )


def _check_transformers_install():
    if _transformers_import_error is not None:
        import os

        if os.getenv("NM_NO_AUTOINSTALL_TRANSFORMERS", False):
            _LOGGER.warning(
                "Unable to import transformers, skipping auto installation "
                "due to NM_NO_AUTOINSTALL_TRANSFORMERS"
            )
            # skip any further checks
            return
        else:
            _LOGGER.warning(
                "deepsparse-transformers installation not detected. Installing "
                "deepsparse-transformers dependencies if transformers is already "
                "installed in the environment, it will be overwritten. Set "
                "environment variable NM_NO_AUTOINSTALL_TRANSFORMERS to disable"
            )
            _install_transformers_and_deps()

    # re check import after potential install
    try:
        import transformers as _transformers

        assert _transformers.NM_INTEGRATED
    except Exception:
        _LOGGER.warning(
            "the neuralmagic fork of transformers may not be installed. it can be "
            "installed via "
            f"`pip install {_NM_TRANSFORMERS_NIGHTLY}`"
        )


_check_transformers_install()


from .helpers import *
from .loaders import *
from .pipelines import *
