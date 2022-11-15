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


"""
Pipeline implementation and pydantic models for embedding extraction transformers
tasks
"""


from typing import List, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.log import get_main_logger
from deepsparse.transformers.helpers import truncate_transformer_onnx_model


__all__ = [
    "EmbeddingExtractionOutput",
    "EmbeddingExtractionPipeline",
]

_LOGGER = get_main_logger()


class EmbeddingExtractionOutput(BaseModel):
    """
    Schema for embedding_extraction pipeline output. Values are in batch order
    """

    embeddings: Union[List[List[float]], List[numpy.ndarray]] = Field(
        description="The output of the model which is an embedded "
        "representation of the input"
    )

    class Config:
        arbitrary_types_allowed = True


@Pipeline.register(
    task="embedding_extraction",
    task_aliases=[],
)
class EmbeddingExtractionPipeline(Pipeline):
    """
    LIFECYCLE:

    Initialize:
        1. create base pipeline stopping before engine initialization
        2. extract onnx file path from base pipeline and truncate
        3. finish base pipeline engine initialization

    __call__:
        1. input schema defers to base pipeline
        2. pre-processing deferes to base pipeline
        3. engine_forward runs truncated model
        4. post processing runs embedding aggregation
        5. output schema contains the possibly aggregated embeddings
    """

    def __init__(
        self,
        *,
        base_task: str,
        emb_extraction_layer: Union[int, str, None] = -1,
        model_size: int = 768,
        return_numpy: bool = True,
        **base_pipeline_args,
    ):
        self._base_task = base_task
        self._emb_extraction_layer = emb_extraction_layer
        self._model_size = model_size
        self._return_numpy = return_numpy

        self.base_pipeline = Pipeline.create(
            task=base_task,
            _delay_engine_initialize=True,  # engine initialized after model truncate
            **base_pipeline_args,
        )

        pipeline_keyword_names = {  # TODO: change to inspect
            "model_path",
            "engine_type",
            "batch_size",
            "num_cores",
            "scheduler",
            "input_shapes",
            "alias",
            "context",
            "executor",
            "_delay_engine_initialize",
        }

        pipeline_kwargs = {
            key: base_pipeline_args[key]
            for key in pipeline_keyword_names
            if key in base_pipeline_args
        }

        self._temp_model_directory = None

        super().__init__(**pipeline_kwargs)

        self.base_pipeline.onnx_file_path = self.onnx_file_path
        self.base_pipeline._initialize_engine()

    @property
    def base_task(self) -> str:
        """
        :return: base task to extract embeddings of
        """
        return self._base_task

    @property
    def emb_extraction_layer(self) -> Union[int, str, None]:
        """
        :return: index or name of layer to extract embeddings at, if None
            outputs of full model are treated as model embeddings
        """
        return self._emb_extraction_layer

    def return_numpy(self) -> bool:
        """
        :return: if True returns embeddings as numpy arrays, uses lists of=
            floats otherwise
        """
        return self._return_numpy

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return self.base_pipeline.input_schema

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return EmbeddingExtractionOutput

    def setup_onnx_file_path(self) -> str:
        """
        Performs setup done in pipeline parent class as well as truncating the
        model to an intermediate layer for embedding extraction

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path = self.base_pipeline.onnx_file_path

        if self._emb_extraction_layer is not None:
            onnx_path, self._temp_model_directory = truncate_onnx_embedding_model(
                onnx_path, emb_extraction_layer=self._emb_extraction_layer
            )
        else:
            _LOGGER.info("EmbeddingExtractionPipeline - skipping model truncation")

        return onnx_path

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, either a input_schema object,
            a string text, or a list of inputs
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_schema`
            schema if necessary. If an instance of the `input_schema` is provided
            it will be returned
        """
        return self.base_pipeline.parse_inputs(*args, **kwargs)

    def process_inputs(self, inputs: BaseModel) -> List[numpy.ndarray]:
        """
        Tokenizes input

        :param inputs: inputs to the pipeline.
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        return self.base_pipeline.process_inputs(inputs)

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **_,  # do not remove - these kwargs will be propagated from base pipeline
    ) -> BaseModel:
        """
        Returns raw embedding outputs from model forward pass

        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        if not self.return_numpy:
            engine_outputs = [array.tolist() for array in engine_outputs]
        return self.output_schema(embeddings=engine_outputs)
