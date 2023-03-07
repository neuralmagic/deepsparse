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

import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
import onnx
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = ["TextGenerationPipeline"]


class TextGenerationInput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    data: Dict[str, Any]
    cache: Optional[Dict[str, Any]]


class TextGenerationOutput(BaseModel):
    """
    Schema for inputs to text_classification pipelines
    """

    generated_str: str
    next_token: int
    cache: Dict[str, Any]


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(TransformersPipeline):
    """
    Transformers text generation pipeline.
    It takes a (batch of) text input of any length and generates a text output.
    The maximum length of the generated sequence is determined by the attribute
    `maximum_generation_length` of the model. The generation process terminate earlier
    if all the sequences in the batch terminate with the end of sequence token.

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```
    example usage:
    ```python
    sequences = [["def hello_world():"], ["class FinancialPipeline(Pipeline):"]]
    OR
    sequences = ["def hello_world():", "class FinancialPipeline(Pipeline):"]
    OR
    sequences = "def hello_world():"

    out = pipeline(sequences = sequences)
    print(out.sequences)
    ```
    :param maximum_generation_length: The maximum length of the generated sequence.
        This means that the length of the resulting output will be a sum of the
        length of the input and the maximum_generation_length.
        The default is set to 256.
    """

    def __init__(
        self,
        *,
        maximum_generation_length: int = 258,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.maximum_generation_length = maximum_generation_length

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        pydantic schema for inputs to this pipeline
        """
        return TextGenerationInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        pydantic schema for outputs from this pipeline
        """
        return TextGenerationOutput

    def __call__(self, *args, **kwargs) -> BaseModel:
        """
        The main entry point for the pipeline. It runs
        prediction of the next token in the autoregressive
        fashion, with the possibility of earlier termination
        when end of sequence tokens are encountered for all the batches
        """

        return super().__call__(*args, **kwargs)

    def _process_token_inputs(self, inputs):
        tokens = inputs.data
        engine_input = self.tokens_to_engine_input(tokens)
        return engine_input

    def process_inputs(
        self, inputs: BaseModel
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the
            TextGenerationInput
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """

        tokens = inputs.data
        cache = inputs.cache
        if cache:
            cache_ = dict()
            for k_old, v in cache.items():
                k_new = k_old
                if k_new.startswith("present"):
                    k_new = k_new.replace("present", "past_key_values")
                cache_[k_new] = v
            tokens.update(cache_)
        engine_input = self.tokens_to_engine_input(tokens)

        return engine_input

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :param engine_outputs: outputs of the pipeline engine. Must be a list of
            numpy arrays
        :param kwargs: additional keyword arguments to pass to the pipeline
        :return: outputs of this model embedded in a TextGenerationOutput object
        """
        assert len(engine_outputs) == 41

        model = onnx.load(os.path.join(self.model_path, "model.onnx"))
        onnx_output_names = [x.name for x in model.graph.output]
        output_dict = {
            name: arr for name, arr in zip(onnx_output_names, engine_outputs)
        }

        kv_cache = {k: v for k, v in output_dict.items() if k.startswith("present")}
        next_token = numpy.argmax(output_dict["logits"], axis=2)[0][-1]
        generated_str = self.tokenizer.decode(next_token, skip_special_tokens=True)

        return TextGenerationOutput(
            generated_str=generated_str, next_token=next_token, cache=kv_cache
        )

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[TransformersPipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        tokenizer = pipelines[0].tokenizer
        tokens = tokenizer(
            input_schema.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)
