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

# postprocessing adapted from huggingface/transformers

# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Double check this
"""
Pipeline implementation and pydantic models for embedding extraction transformers
tasks
"""


from typing import List, Type, Union

import os
import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "EmbeddingExtractionInput",
    "EmbeddingExtractionOutput",
    "EmbeddingExtractionPipeline",
]

CACHE_DIR = os.path.expanduser(os.path.join("~", ".cache", "deepsparse"))


class EmbeddingExtractionInput(BaseModel):
    """
    Schema for inputs to embedding_extraction pipelines
    """

    sequences: Union[List[str], str] = Field(
        description="A string or list of strings representing input to"
        "the embedding_extraction task"
    )


class EmbeddingExtractionOutput(BaseModel):
    """
    Schema for embedding_extraction pipeline output. Values are in batch order
    """

    embeddings: Union[List[numpy.ndarray], numpy.ndarray] = Field(
        description="The output of the model which is an embedded "
        "representation of the input"
    )


@Pipeline.register(
    task="embedding_extraction",
    task_aliases=[],
    default_model_path=(
        "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
        "wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni"
    ),
)
class EmbeddingExtractionPipeline(TransformersPipeline):
    def __init__(
        self,
        *,
        show_progress_bar: bool = False,
        **kwargs,
    ):
        self._show_progress_bar = show_progress_bar

        super().__init__(**kwargs)


    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return EmbeddingExtractionInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return EmbeddingExtractionOutput

    # TODO: Wtf is this
    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[TransformersPipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        current_seq_len = len(input_schema.question.split())

        for pipeline in pipelines:
            if pipeline.sequence_length > current_seq_len:
                return pipeline
        return pipelines[-1]

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        print(args)
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1:
                # passed input_schema schema directly
                if isinstance(args[0], self.input_schema):
                    return args[0]
                return self.input_schema(sequences=args[0])
            else:
                return self.input_schema(sequences=args)

        return self.input_schema(**kwargs)


    def process_inputs(self, inputs: EmbeddingExtractionInput) -> List[numpy.ndarray]:
        tokens = pipeline.tokenizer(
            inputs.sequences,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )

        '''
        postprocessing_kwargs = dict(
            sequences=sequences,
            labels=labels,
            multi_class=multi_class,
        )
        '''
        return self.tokens_to_engine_input(tokens)

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:

        '''
        def _engine_forward(batch_index: int, batch_origin: int):
            labelwise_inputs = engine_inputs_numpy[
                :, batch_origin : batch_origin + pipeline._batch_size, :
            ]
            labelwise_inputs = [labelwise_input for labelwise_input in labelwise_inputs]
            engine_output = pipeline.engine(labelwise_inputs)
            engine_outputs[batch_index] = engine_output

        engine_inputs_numpy = numpy.array(engine_inputs)
        engine_outputs = [
            None for _ in range(engine_inputs_numpy.shape[1] // pipeline._batch_size)
        ]

        futures = [
            pipeline._thread_pool.submit(_engine_forward, batch_index, batch_origin)
            for batch_index, batch_origin in enumerate(
                range(0, engine_inputs_numpy.shape[1], pipeline._batch_size)
            )
        ]
        wait(futures)
        '''
        return self.engine(engine_inputs)

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        return EmbeddingExtractionOutput(engine_outputs)
