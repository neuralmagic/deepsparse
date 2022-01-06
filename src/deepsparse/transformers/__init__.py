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

try:
    import transformers as _transformers

    transformers_import_error = None
except Exception as transformers_import_err:
    transformers_import_error = transformers_import_err


def _check_transformers_install():
    if transformers_import_error is None:
        return
    raise ImportError(
        "No installation of transformers found. It is recommended to use the "
        "sparseml fork of transformers which can be installed under "
        "git+https://github.com/neuralmagic/transformers.git or sparseml[transformers]"
    )


_check_transformers_install()


from .helpers import *
from .pipelines import *
