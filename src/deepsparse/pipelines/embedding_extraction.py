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
import inspect
from typing import Any, List, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.log import get_main_logger
from deepsparse.utils import truncate_onnx_embedding_model


__all__ = [
    "EmbeddingExtractionOutput",
    "EmbeddingExtractionPipeline",
]


_LOGGER = get_main_logger()


class EmbeddingExtractionOutput(BaseModel):
    """
    Schema for embedding_extraction pipeline output. Values are in batch order
    """

    # List[Any] is for accepting numpy arrays
    embeddings: Union[List[List[float]], List[Any]] = Field(
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
    embedding extraction pipeline for extracting intermediate layer embeddings

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param emb_extraction_layer: if an int, the layer number from which
        the embeddings will be extracted. If a string, the name of last
        ONNX node in model to draw embeddings from. If None, leave the model
        unchanged. Default is None
    :param flatten_outputs: if True, embeddings will be flattened along the batch (0)
        dimension. Default False
    :param return_numpy: return embeddings a list of numpy arrays, list of lists
        of floats otherwise. Default is False
    """

    def __init__(
        self,
        *,
        base_task: str,
        emb_extraction_layer: Union[int, str, None] = None,
        return_numpy: bool = False,  # to support Pydantic Validation
        flatten_outputs: bool = False,
        **base_pipeline_args,
    ):
        self._base_task = base_task
        self._emb_extraction_layer = emb_extraction_layer
        self._return_numpy = return_numpy
        self._flatten_outputs = flatten_outputs

        # initialize engine after model truncate
        self.base_pipeline = Pipeline.create(
            task=base_task,
            _delay_engine_initialize=True,
            **base_pipeline_args,
        )

        pipeline_keyword_names = inspect.signature(Pipeline).parameters.keys()
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

    @property
    def return_numpy(self) -> bool:
        """
        :return: if True, embeddings will be flattened along the batch (0)
            dimension
        """
        return self._return_numpy

    @property
    def flatten_outputs(self) -> bool:
        """
        :return: if True returns embeddings as numpy arrays, uses lists of=
            floats otherwise
        """
        return self._flatten_outputs

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
        if self.flatten_outputs:
            engine_outputs = [
                array.reshape(array.shape[0], -1) for array in engine_outputs
            ]
        if not self.return_numpy:
            engine_outputs = [array.tolist() for array in engine_outputs]
        return self.output_schema(embeddings=engine_outputs)
