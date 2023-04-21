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
import time
from typing import List, Union

from deepsparse import Pipeline
from deepsparse.open_pif_paf.utils.validation.helpers import deepsparse_fields_to_torch
from openpifpaf.decoder import CifCaf
from openpifpaf.decoder.decoder import DummyPool


LOG = logging.getLogger(__name__)

__all__ = ["DeepSparseCifCaf"]


class DeepSparseCifCaf(CifCaf):
    def __init__(
        self,
        head_metas: List[Union["Cif", "Caf"]],  # noqa: F821
        pipeline: Pipeline,
    ):
        self.pipeline = pipeline
        cif_metas, caf_metas = head_metas
        super().__init__([cif_metas], [caf_metas])

    # adapted from OPENPIFPAF GITHUB:
    # https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/decoder/decoder.py
    # the appropriate edits are marked with # deepsparse edit: <edit comment>

    # deepsparse edit: removed model argument (not needed, substituted with '_')
    def batch(self, _, image_batch, *, device=None, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        # deepsparse edit: inference using deepsparse pipeline
        # instead of torch model
        fields_batch = deepsparse_fields_to_torch(
            self.pipeline(images=image_batch.numpy())
        )
        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug("parallel execution with worker %s", self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch)
        )
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug(
            "time: nn = %.1fms, dec = %.1fms",
            self.last_nn_time * 1000.0,
            self.last_decoder_time * 1000.0,
        )
        return result
