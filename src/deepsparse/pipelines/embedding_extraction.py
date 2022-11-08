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


from enum import Enum
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


class ExtractionStrategy(str, Enum):
    """
    Schema for supported extraction strategies
    """

    per_token = "per_token"
    reduce_mean = "reduce_mean"
    reduce_max = "reduce_max"
    cls_token = "cls_token"

    @classmethod
    def to_list(cls) -> List[str]:
        return cls._value2member_map_


@Pipeline.register(
    task="embedding_extraction",
    task_aliases=[],
    default_model_path="",  # TODO determine what goes here
)
class EmbeddingExtractionPipeline(Pipeline):
    """
    LIFECYCLE:

    Initialize:
        1. create base pipeline stopping before engine initailization
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
        extraction_strategy: ExtractionStrategy = "per_token",
        return_numpy: bool = True,
        **base_pipeline_args,
    ):
        self._base_task = base_task
        self._emb_extraction_layer = emb_extraction_layer
        self._model_size = model_size
        self._extraction_strategy = extraction_strategy
        self._return_numpy = return_numpy

        if self._extraction_strategy not in ExtractionStrategy.to_list():
            raise ValueError(
                f"Unsupported extraction_strategy {self._extraction_strategy}"
            )

        self.base_pipeline = Pipeline.create(
            task=base_task,
            _delay_engine_initialize=True,  # engine initialized after model truncate
            **base_pipeline_args,
        )

        # TODO @rahul-tuli: IMPORTANT - extract keyword arg keys from deepsparse.pipeline
        # should only incldue these out of **base_pipeline_args
        # keeping in kwargs for now so it will definitely fail until we do this
        super().__init__(**kwargs)

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
    def extraction_strategy(self) -> ExtractionStrategy:
        """
        :return: aggregation method used for final embeddings
        """
        return self._extraction_strategy

    @property
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

        # TODO: use truncate_onnx_model, not truncate_transformer_onnx_model
        # potentially find a way to keep transformers specific helpers
        # basically want to copy the UX here so we can push
        if self._emb_extraction_layer is not None:
            (
                onnx_path,
                self.onnx_output_names,
                self._temp_model_directory,
            ) = truncate_transformer_onnx_model(
                onnx_path,
                emb_extraction_layer=self._emb_extraction_layer,
                hidden_layer_size=self._model_size,
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
        pad_masks: numpy.ndarray,
        cls_masks: numpy.ndarray,
    ) -> BaseModel:
        """
        Implements extraction_strategy from the intermediate layer and returns its value

        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :param pad_masks: mask of the padding token for each engine input
        :param cls_masks: mask of the cls token for each engine input
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        if isinstance(engine_outputs, list):
            engine_outputs = engine_outputs[0]

        embeddings = []
        assert len(engine_outputs) == len(pad_masks) == len(cls_masks)
        for engine_output, pad_mask, cls_mask in zip(
            engine_outputs, pad_masks, cls_masks
        ):
            # extraction strategy
            if self._extraction_strategy == ExtractionStrategy.per_token:
                embedding = engine_output
            if self._extraction_strategy == ExtractionStrategy.reduce_mean:
                masked_output = self._remove_1d_mask(
                    engine_output, mask=(pad_mask | cls_mask)
                )
                embedding = masked_output.mean(axis=0)
            if self._extraction_strategy == ExtractionStrategy.reduce_max:
                masked_output = self._remove_1d_mask(
                    engine_output, mask=(pad_mask | cls_mask)
                )
                embedding = masked_output.max(axis=0)
            if self._extraction_strategy == ExtractionStrategy.cls_token:
                embedding = engine_output[numpy.where(cls_mask)[0][0]]

            # flatten
            embedding = embedding.flatten()

            if not self._return_numpy:
                embedding = embedding.tolist()

            embeddings.append(embedding)

        return self.output_schema(embeddings=embeddings)

    def _remove_1d_mask(
        self, array: numpy.ndarray, mask: numpy.ndarray
    ) -> numpy.ndarray:
        # Helper function to mask out values from a 1 dimensional mask

        # :param array: array containing values to be masked out
        # :param mask: 1 dimensional mask
        # :return: numpy masked array
        array_masked = numpy.ma.masked_array(array)
        array_masked[mask] = numpy.ma.masked

        return array_masked
