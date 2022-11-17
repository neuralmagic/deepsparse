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


from enum import Enum
from typing import Any, List, Type, Union

import numpy
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.log import get_main_logger
from deepsparse.transformers.helpers import truncate_transformer_onnx_model
from deepsparse.transformers.pipelines import TransformersPipeline


__all__ = [
    "EmbeddingExtractionInput",
    "EmbeddingExtractionOutput",
    "TransformersEmbeddingExtractionPipeline",
]

_LOGGER = get_main_logger()


class EmbeddingExtractionInput(BaseModel):
    """
    Schema for inputs to transformers_embedding_extraction pipelines
    """

    inputs: Union[str, List[str]] = Field(
        description="A list of sequences from which to get embeddings"
    )


class EmbeddingExtractionOutput(BaseModel):
    """
    Schema for transformers_embedding_extraction pipeline output.
    Values are in batch order
    """

    # List[Any] is for accepting numpy arrays
    embeddings: Union[List[List[float]], List[Any]] = Field(
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
    task="transformers_embedding_extraction",
    task_aliases=[],
    default_model_path=(
        "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
        "wikipedia_bookcorpus/pruned80_quant-none-vnni"
    ),
)
class TransformersEmbeddingExtractionPipeline(TransformersPipeline):
    """
    embedding extraction pipeline for extracting intermediate layer embeddings
    from transformer models

    example instantiation:
    ```python
    transformers_embedding_extraction_pipeline = Pipeline.create(
        task="transformers_embedding_extraction",
        model_path="masked_language_modeling_model_dir/",
    )
    results = transformers_embedding_extraction_pipeline(
        [
            "the warriors have won the nba finals"
            "the warriors are the greatest basketball team ever"
        ]
    )
    emb_1, emb_2 = results.embeddings
    # (expect emb_1 and emb_2 to have high cosine similiarity)
    ```

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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param emb_extraction_layer: if an int, the transformer layer number from
        which the embeddings will be extracted. If a string, the name of last
        ONNX node in model to draw embeddings from. If None, leave the model
        unchanged. Default is -1 (last transformer layer before prediction head)
    :param model_size: size of transformer model (size of hidden layer per token
        if the model is cut). Default is 768
    :param extraction_strategy: method of pooling embedding values. Currently
        supported values are 'per_token', 'reduce_mean', 'reduce_max' and 'cls_token'.
        Default is 'per_token'
    :param return_numpy: return embeddings a list of numpy arrays, list of lists
        of floats otherwise. Default is False
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels. Default is None
    """

    def __init__(
        self,
        *,
        emb_extraction_layer: Union[int, str, None] = -1,
        model_size: int = 768,
        extraction_strategy: ExtractionStrategy = "per_token",
        return_numpy: bool = False,  # to support Pydantic Validation
        **kwargs,
    ):
        self._emb_extraction_layer = emb_extraction_layer
        self._model_size = model_size
        self._extraction_strategy = extraction_strategy
        self._return_numpy = return_numpy

        if self._extraction_strategy not in ExtractionStrategy.to_list():
            raise ValueError(
                f"Unsupported extraction_strategy {self._extraction_strategy}"
            )

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

    def setup_onnx_file_path(self) -> str:
        """
        Performs setup done in pipeline parent class as well as truncating the
        model to an intermediate layer for embedding extraction

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path = super().setup_onnx_file_path()

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
            _LOGGER.info("Skipping model truncation")

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
        if args and kwargs:
            raise ValueError(
                f"{self.__class__} only support args OR kwargs. Found "
                f" {len(args)} args and {len(kwargs)} kwargs"
            )

        if not args:
            return self.input_schema(**kwargs)
        if isinstance(args, str):
            return self.input_schema(inputs=[args[0]])
        if len(args) != 1:
            return self.input_schema(inputs=args)
        if isinstance(args[0], self.input_schema):
            return args[0]
        return self.input_schema(inputs=args[0])

    def process_inputs(self, inputs: EmbeddingExtractionInput) -> List[numpy.ndarray]:
        """
        Tokenizes input

        :param inputs: inputs to the pipeline.
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        if isinstance(inputs.inputs, str):
            inputs.inputs = [inputs.inputs]

        # tokenization matches https://github.com/texttron/tevatron
        tokens = self.tokenizer(
            inputs.inputs,
            add_special_tokens=True,
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
            return_tensors="np",
        )

        # mask padding and cls_token
        pad_masks = tokens["input_ids"] == self.tokenizer.pad_token_id
        cls_masks = tokens["input_ids"] == self.tokenizer.cls_token_id

        return self.tokens_to_engine_input(tokens), {
            "pad_masks": pad_masks,
            "cls_masks": cls_masks,
        }

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

    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        tokenizer = pipelines[0].tokenizer
        tokens = tokenizer(
            input_schema.inputs,
            add_special_tokens=True,
            return_tensors="np",
            padding=False,
            truncation=False,
        )
        input_seq_len = max(map(len, tokens["input_ids"]))
        return TransformersPipeline.select_bucket_by_seq_len(input_seq_len, pipelines)

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
