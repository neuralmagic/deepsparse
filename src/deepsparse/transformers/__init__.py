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

from deepsparse.analytics import deepsparse_analytics as _analytics


_analytics.send_event("python__transformers__init")


_LOGGER = _logging.getLogger(__name__)


def _check_transformers_install():
    import transformers as _transformers

    if not _transformers.NM_INTEGRATED:
        _LOGGER.warning(
            "the neuralmagic fork of transformers may not be installed. it can be "
            f"installed via `pip install {nm_transformers}`"
        )


_check_transformers_install()


from .helpers import *
from .loaders import *
from .pipelines import *
