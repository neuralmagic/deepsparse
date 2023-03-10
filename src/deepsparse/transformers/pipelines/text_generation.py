from typing import Type, List, Union, Tuple, Dict, Any, Optional

import numpy
from pydantic import BaseModel, Field
from transformers import BatchEncoding
from deepsparse import Pipeline
from deepsparse.transformers.pipelines.helper import AutoregressivePipeline



__all__ = ["TextGenerationPipeline"]

class InputSchema(BaseModel):
    sequences: Union[str, List[str]]

class OutputSchema(BaseModel):
    sequences: Union[str, List[str]]



@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(AutoregressivePipeline):

    @staticmethod
    def route_input_to_bucket(*args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs) -> Pipeline:
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray], **kwargs) -> BaseModel:

        return None

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return InputSchema


    @property
    def output_schema(self) -> Type[BaseModel]:
        return OutputSchema


    def process_inputs(
            self,
            inputs: BaseModel):

        sequences = inputs.sequences
        if isinstance(sequences, List) and all(
                isinstance(sequence, List) and len(sequence) == 1 for sequence in sequences
        ):
            # if batch items contain only one sequence but are wrapped in lists, unwrap
            # for use as tokenizer input
            sequences = [sequence[0] for sequence in sequences]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_tokens = self.tokenizer(
            sequences,
            add_special_tokens=True,
            return_tensors="np",
            max_length=self.sequence_length,
            padding="max_length")


        onnx_input_names = [input_name for input_name in self.onnx_input_names if not input_name.startswith("past_key_values")]
        engine_input = self.tokens_to_engine_input(input_tokens, onnx_input_names = onnx_input_names)

        # a boolean mask that indicates which tokens are valid (are non-padding tokens)
        valid_tokens_mask = numpy.where(
            engine_input[0] == self.tokenizer.pad_token_id, 1, 0
        )

        preprocessing_kwargs = dict(
            input_sequence=engine_input[0], valid_tokens_mask=valid_tokens_mask
        )

        return engine_input, preprocessing_kwargs




