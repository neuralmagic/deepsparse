import time
import torch
import logging
from openpifpaf.decoder import CifCaf
from openpifpaf.decoder.decoder import DummyPool
from typing import List
from deepsparse import Pipeline

from deepsparse.open_pif_paf.utils.validation.helpers import deepsparse_fields_to_torch

LOG = logging.getLogger(__name__)

class DeepSparseCifCaf(CifCaf):
    def __init__(self, pipeline: Pipeline, head_metas: List[None]):
        self.pipeline = pipeline
        cif_metas, caf_metas = head_metas
        super().__init__([cif_metas], [caf_metas])

    # adapted from OPENPIFPAF GITHUB:
    # https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/decoder/decoder.py
    # the appropriate edits are marked with # deepsparse edit: <edit comment>
    def batch(self, model, image_batch, *, device=None, gt_anns_batch=None):
        """From image batch straight to annotations batch."""
        start_nn = time.perf_counter()
        fields_batch = self.fields_batch(model, image_batch, device=device)
        fields_batch = deepsparse_fields_to_torch(self.pipeline(images=image_batch.numpy()))
        self.last_nn_time = time.perf_counter() - start_nn

        if gt_anns_batch is None:
            gt_anns_batch = [None for _ in fields_batch]

        if not isinstance(self.worker_pool, DummyPool):
            # remove debug_images to save time during pickle
            image_batch = [None for _ in fields_batch]
            gt_anns_batch = [None for _ in fields_batch]

        LOG.debug('parallel execution with worker %s', self.worker_pool)
        start_decoder = time.perf_counter()
        result = self.worker_pool.starmap(
            self._mappable_annotations, zip(fields_batch, image_batch, gt_anns_batch))
        self.last_decoder_time = time.perf_counter() - start_decoder

        LOG.debug('time: nn = %.1fms, dec = %.1fms',
                  self.last_nn_time * 1000.0,
                  self.last_decoder_time * 1000.0)
        return result

