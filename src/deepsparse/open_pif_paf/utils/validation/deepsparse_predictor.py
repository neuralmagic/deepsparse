import logging
from openpifpaf import Predictor
from deepsparse import Pipeline
from deepsparse.open_pif_paf.utils.validation.deepsparse_decoder import DeepSparseCifCaf

LOG = logging.getLogger(__name__)

# adapted from OPENPIFPAF GITHUB:
# https://github.com/openpifpaf/openpifpaf/blob/main/src/openpifpaf/predictor.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
class DeepSparsePredictor(Predictor):
    """Convenience class to predict from various inputs with a common configuration."""
    # deepsparse edit: allow for passing in a pipeline
    def __init__(self, pipeline: Pipeline, **kwargs):
        super().__init__(**kwargs)
        # deepsparse edit: allow for passing in a pipeline and fix the processor
        # to CifCaf processor
        self.processor = DeepSparseCifCaf(pipeline = pipeline, head_metas = self.model_cpu.head_metas)


