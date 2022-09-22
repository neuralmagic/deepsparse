"""
GPT pipeline
"""
from typing import Any, Dict, List, Tuple, Type, Union

import numpy
from deepsparse import Pipeline

__all__ = [
    "GPTPipeline"
]

from pydantic import BaseModel
from transformers import GPT2Tokenizer


class GPTInput(BaseModel):
    prompt: str


class GPTOutput(BaseModel):
    output: Any


@Pipeline.register(
    task="gpt2",
)
class GPTPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def setup_onnx_file_path(self) -> str:
        return self.model_path

    def process_inputs(self, inputs: BaseModel) -> Union[
        List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        model_input = self.tokenizer(inputs.prompt, return_tensors="np")
        return list(model_input.values())

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        outputs = engine_outputs
        if isinstance(outputs, list):
            outputs = outputs[0]

        scores = (
            numpy.exp(outputs) / numpy.exp(outputs).sum(-1, keepdims=True)
        )

        tokenids = [score.argmax() for score in scores]
        tokens = self.tokenizer.batch_decode(tokenids)
        return GPTOutput(output=tokens)

    @property
    def input_schema(self) -> Type[BaseModel]:
        return GPTInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        return GPTOutput



