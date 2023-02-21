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

import logging

from deepsparse import Pipeline
from deepsparse.open_pif_paf.utils.validation.deepsparse_decoder import DeepSparseCifCaf
from openpifpaf import Predictor


LOG = logging.getLogger(__name__)

__all__ = ["DeepSparsePredictor"]


# adapted from OPENPIFPAF GITHUB:
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/predictor.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
class DeepSparsePredictor(Predictor):
    """
    Convenience class to predict from various
    inputs with a common configuration.
    """

    # deepsparse edit: allow for passing in a pipeline
    def __init__(self, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        # deepsparse edit: allow for passing in a pipeline and fix the processor
        # to CifCaf processor. Note: we are creating here a default torch model
        # but we only use it to get its head metas. This is required to
        # initialize the DeepSparseCifCaf processor.
        self.processor = DeepSparseCifCaf(
            pipeline=pipeline, head_metas=self.model_cpu.head_metas
        )
