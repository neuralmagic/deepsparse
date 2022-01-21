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

    _transformers_import_error = None
except Exception as _transformers_import_err:
    _transformers_import_error = _transformers_import_err


_LOGGER = _logging.getLogger(__name__)


def _install_transformers_and_deps():

    import deepsparse as _deepsparse
    import pip as _pip

    transformers_branch = (
        "master"
        if not _deepsparse.is_release
        else f"release/{_deepsparse.version.version_major_minor}"
    )
    transformers_requirement = (
        "transformers @ git+https://github.com/neuralmagic/transformers.git"
        f"@{transformers_branch}"
    )

    try:
        _pip.main(
            [
                "install",
                transformers_requirement,
                "datasets",
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
            "`pip install git+https://github.com/neuralmagic/transformers.git`"
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
            _LOGGER.info(
                "No installation of transformers found. Installing deepsparse-transformers "
                "dependencies"
            )
            _install_transformers_and_deps()

    # check NM fork installed with QATMatMul available
    try:
        import transformers as _transformers

        _transformers.models.bert.modeling_bert.QATMatMul
    except Exception:
        _LOGGER.warning(
            "transformers.models.bert.modeling_bert.QATMatMul not availalbe. the"
            "neuralmagic fork of transformers may not be installed. it can be "
            "installed via "
            "`pip install git+https://github.com/neuralmagic/transformers.git`"
        )


_check_transformers_install()


from .helpers import *
from .loaders import *
from .pipelines import *
from .server import *
