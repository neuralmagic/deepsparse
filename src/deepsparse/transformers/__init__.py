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
from deepsparse.auto_install import auto_pip_install, Dependency
import deepsparse


_DEEPSPARSE_VERSION = (
    f"v{deepsparse.version.version_major_minor}" if deepsparse.is_release else "nightly"
)

_NM_TRANSFORMERS = (
    "https://github.com/neuralmagic/transformers/releases/download/"
    f"{_DEEPSPARSE_VERSION}/transformers-4.18.0.dev0-py3-none-any.whl"
)


auto_pip_install(
    __qualname__,
    Dependency(
        _NM_TRANSFORMERS,
        import_name="transformers",
        requirements=["datasets<=1.18.4", "sklearn", "seqeval"],
        check_attr="NM_INTEGRATED",
    ),
)


from .helpers import *
from .loaders import *
from .pipelines import *

__all__ = helpers.__all__ + loaders.__all__ + pipelines.__all__
