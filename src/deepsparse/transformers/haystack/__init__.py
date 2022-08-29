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
Helpers for running haystack pipelines and nodes with DeepSparse and integrating
deepset-ai/haystack
"""

# flake8: noqa
# isort: skip_file


import logging as _logging
import os as _os

import deepsparse as _deepsparse


_HAYSTACK_PREFERRED_VERSION = "1.4.0"
_HAYSTACK_EXTRAS = "[all]"


# check haystack installation
try:
    import haystack as _haystack

    if _haystack.__version__ != _HAYSTACK_PREFERRED_VERSION:
        raise ValueError(
            f"Deepsparse requires farm-haystack=={_HAYSTACK_PREFERRED_VERSION}, "
            f"but found {_haystack.__version__}"
        )
    _haystack_import_error = None
except Exception as _haystack_import_err:
    _haystack_import_error = _haystack_import_err

_LOGGER = _logging.getLogger(__name__)


def _install_haystack_and_deps():
    import subprocess as _subprocess
    import sys as _sys

    try:
        _subprocess.check_call(
            [
                _sys.executable,
                "-m",
                "pip",
                "install",
                f"farm-haystack{_HAYSTACK_EXTRAS}=={_HAYSTACK_PREFERRED_VERSION}",
                "--no-dependencies",
            ]
        )

        import haystack as _haystack

        _LOGGER.info("haystack and dependencies successfully installed")
    except Exception:
        raise ValueError(
            "Unable to install and import haystack dependencies. Check "
            "that haystack is installed, if not, install via "
            "`pip install deepsparse[haystack]` and `pip install "
            f"farm-haystack{_HAYSTACK_EXTRAS}=={_HAYSTACK_PREFERRED_VERSION} "
            "--no-dependencies`"
        )


def _check_haystack_install():
    if _haystack_import_error is not None:
        import os

        if os.getenv("NM_NO_AUTOINSTALL_HAYSTACK", False):
            _LOGGER.warning(
                "Unable to import haystack, skipping auto installation "
                "due to NM_NO_AUTOINSTALL_HAYSTACK"
            )
            # skip any further checks
            return
        else:
            _LOGGER.warning(
                "haystack installation not detected. Installing "
                "haystack dependencies if haystack is already "
                "installed in the environment, it will be overwritten. Set "
                "environment variable NM_NO_AUTOINSTALL_HAYSTACK to disable"
            )
            _install_haystack_and_deps()

    # re check import after potential install
    try:
        import haystack as _haystack

        if _haystack.__version__ != _HAYSTACK_PREFERRED_VERSION:
            raise ValueError(
                f"Deepsparse requires farm-haystack=={_HAYSTACK_PREFERRED_VERSION}, "
                f"but found {_haystack.__version__}"
            )
    except Exception:
        _LOGGER.warning(
            "haystack and its dependencies may not be installed. They can be installed "
            "via `pip install deepsparse[haystack]` and `pip install "
            f"farm-haystack{_HAYSTACK_EXTRAS}=={_HAYSTACK_PREFERRED_VERSION} "
            "--no-dependencies`"
        )


_check_haystack_install()

from .nodes import *
from .pipeline import *
from .helpers import *
