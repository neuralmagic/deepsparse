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

"""
Pipeline implementation and pydantic models for embedding extraction transformers
tasks
"""


import os
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, List, Optional, Tuple, Type, Union

import tqdm
import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.transformers.helpers import cut_transformer_onnx_model
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "EmbeddingExtractionInput",
    "EmbeddingExtractionOutput",
    "EmbeddingExtractionPipeline",
]

class EmbeddingExtractionInput(BaseModel):
    """
    Schema for inputs to embedding_extraction pipelines
    """

    texts: Union[List[str]] = Field(
        description="A list of document contents"
    )


class EmbeddingExtractionOutput(BaseModel):
    """
    Schema for embedding_extraction pipeline output. Values are in batch order
    """

    # TODO: converting to lists leads to slowdowns. is there a better way?
    embeddings: Union[List[List[float]], List[float]] = Field(
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
        return_all_pos: bool = False,
        show_progress_bar: bool = False,
        context: Optional[Context] = None,
        **kwargs,
    ):
        self._return_all_pos = return_all_pos
        self._show_progress_bar = show_progress_bar

        if context is None:
            # num_streams is arbitrarily chosen to be any value >= 2
            context = Context(num_cores=None, num_streams=2)
            kwargs.update({"context": context})

            self._thread_pool = ThreadPoolExecutor(
                max_workers=context.num_streams or 2,
                thread_name_prefix="deepsparse.pipelines.embedding_extraction",
            )

        super().__init__(**kwargs)

    @staticmethod
    def should_bucket(*args, **kwargs) -> bool:
        return False

    @staticmethod
    def create_pipeline_buckets(*args, **kwargs) -> List[Pipeline]:
        pass

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        pass

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

    def setup_onnx_file_path(self) -> str:
        """
        Performs setup done in parent class as well as cutting the model to an
        intermediate layer for latent space comparison

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path = super().setup_onnx_file_path()

        (
            onnx_path,
            self.onnx_output_names,
            self._temp_model_directory,
        ) = cut_transformer_onnx_model(onnx_path)

        return onnx_path

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if args:
            if len(args) == 1:
                if isinstance(args[0], self.input_schema):
                    return args[0]
                return self.input_schema(texts=args[0])
            else:
                return self.input_schema(texts=args)

        return self.input_schema(**kwargs)

    def process_inputs(self, inputs: EmbeddingExtractionInput) -> List[numpy.ndarray]:
        tokens = self.tokenizer(
            inputs.texts,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )

        return self.tokens_to_engine_input(tokens)

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        def _engine_forward(batch_origin: int):
            # run engine
            engine_input = engine_inputs_numpy[
                :, batch_origin : batch_origin + self._batch_size, :
            ]
            engine_input = [model_input for model_input in engine_input]
            engine_output = self.engine(engine_input)

            # get embedding from cls token
            if not self._return_all_pos:
                target_index = 0
            else:
                target_index = range(engine_output[0].shape[1])
            cls_token_embedding = engine_output[0][:, target_index, :]

            # save results
            engine_outputs[batch_origin: batch_origin + self._batch_size] = cls_token_embedding

            # update tqdm
            if progress:
                progress.update(1)

        engine_inputs_numpy = numpy.array(engine_inputs)

        if engine_inputs_numpy.shape[1] % self._batch_size != 0:
            raise ValueError(
                f"number of engine inputs {engine_inputs_numpy.shape[1]} must "
                f"be divisible by batch_size {self._batch_size}"
            )

        engine_outputs = [
            None for _ in range(engine_inputs_numpy.shape[1])
        ]

        num_batches = engine_inputs_numpy.shape[1] // self._batch_size
        progress = tqdm.tqdm(desc="Inferencing Samples", total=num_batches) if self._show_progress_bar else None

        futures = [
            self._thread_pool.submit(_engine_forward, batch_origin)
            for batch_origin in range(0, engine_inputs_numpy.shape[1], self._batch_size)
        ]

        wait(futures)
        #[_engine_forward(batch_origin) for batch_origin in range(0, engine_inputs_numpy.shape[1], self._batch_size)]

        return engine_outputs

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        embeddings = [engine_output.tolist() for engine_output in engine_outputs]
        return self.output_schema(embeddings=embeddings)
