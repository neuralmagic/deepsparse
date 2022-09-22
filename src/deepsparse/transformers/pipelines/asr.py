from typing import Dict, Any, List

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline




class ASRInput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    input: Dict[str, Any] = Field(
        description="{'sampling rate': int, raw_audio: np.ndarray'}"
    )

class ASROutput(BaseModel):
    pass

@Pipeline.register(
    task="asr",
    task_aliases=None,
    default_model_path=None
)
class ASRPipeline(TransformersPipeline):
    pass



