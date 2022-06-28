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


from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Optional, Type, Union

import numpy
import tqdm
from pydantic import BaseModel, Field
from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.transformers.helpers import truncate_transformer_onnx_model
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

    texts: List[str] = Field(
        description="A list of texts from which to get embeddings"
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
    """
    embedding extraction pipeline for extracting intermediate layer embeddings
    from transformer models

    example instantiation:
    ```python
    embedding_extraction_pipeline = Pipeline.create(
        task="embedding_extraction",
        model_path="masked_language_modeling_model_dir/",
    )
    results = embedding_extraction_pipeline(
        [
            "the warriors have won the nba finals"
            "the warriors are the greatest basketball team ever"
        ]
    )
    emb_1, emb_2 = results.embeddings
    (expect emb_1 and emb_2 to have high cosine similiarity)
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
        Default is 128
    :param emb_extraction_layer: transformer layer number from which the embeddings
        will be extracted. Default is -1 (last layer)
    :param model_size: size of transformer model (size of hidden layer per token).
        Default is 768
    :param extraction_strategy: TODO
    :param show_progress_bar: token index(es) to extract from the embedding. Can
        either be an integer representing the CLS token index, or a list of indexes
        to extract. Default is 0
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels. Default is None
    """
    def __init__(
        self,
        *,
        emb_extraction_layer: int = -1,
        model_size: int = 768,
        extraction_strategy: str = "cls_token",
        show_progress_bar: bool = False,
        context: Optional[Context] = None,
        **kwargs,
    ):
        self._emb_extraction_layer = emb_extraction_layer
        self._model_size = model_size
        self._extraction_strategy = extraction_strategy
        self._show_progress_bar = show_progress_bar

        if context is None:
            # num_streams is arbitrarily chosen to be any value >= 2
            context = Context(num_cores=None, num_streams=2)
            kwargs.update({"context": context})

            self._thread_pool = ThreadPoolExecutor(
                max_workers=context.num_streams or 2,
                thread_name_prefix="deepsparse.pipelines.embedding_extraction",
            )

        if self._extraction_strategy not in ["per_token", "reduce_mean", "reduce_max", "cls_token"]:
            raise ValueError(
                f"Unsupported extraction_strategy {self._extraction_strategy}"
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
        Performs setup done in pipeline parent class as well as truncating the
        model to an intermediate layer for embedding extraction

        :return: file path to the processed ONNX file for the engine to compile
        """
        onnx_path = super().setup_onnx_file_path()

        (
            onnx_path,
            self.onnx_output_names,
            self._temp_model_directory,
        ) = truncate_transformer_onnx_model(onnx_path, emb_extraction_layer=self._emb_extraction_layer, model_size=self._model_size)

        return onnx_path

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, either a input_schema object,
            a string text, or a list of texts
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

        if args:
            if isinstance(args, str):
                return self.input_schema(texts=[args[0]])

            if len(args) == 1:
                if isinstance(args[0], self.input_schema):
                    return args[0]
                else:
                    return self.input_schema(texts=args[0])
            else:
                return self.input_schema(texts=args)
        else:
            return self.input_schema(**kwargs)

    def process_inputs(self, inputs: EmbeddingExtractionInput) -> List[numpy.ndarray]:
        """
        Tokenizes input texts

        :param inputs: inputs to the pipeline.
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine
        """
        tokens = self.tokenizer(
            inputs.texts,
            add_special_tokens=True,
            return_tensors="np",
            padding=PaddingStrategy.MAX_LENGTH.value,
            truncation=TruncationStrategy.LONGEST_FIRST.value,
        )

        return self.tokens_to_engine_input(tokens)

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Handles multithreaded engine inference

        :param engine_inputs: list of numpy inputs to Pipeline engine forward pass
        :return: result of forward pass to Pipeline engine
        """
        def _engine_forward(batch_origin: int):
            # run engine
            engine_input = engine_inputs_numpy[
                :, batch_origin : batch_origin + self._batch_size, :
            ]
            engine_input = [model_input for model_input in engine_input]
            engine_output = self.engine(engine_input)

            # save results
            engine_outputs[
                batch_origin : batch_origin + self._batch_size
            ] = engine_output[0]

            # update tqdm
            if progress:
                progress.update(1)

        engine_inputs_numpy = numpy.array(engine_inputs)

        if engine_inputs_numpy.shape[1] % self._batch_size != 0:
            raise ValueError(
                f"number of engine inputs {engine_inputs_numpy.shape[1]} must "
                f"be divisible by batch_size {self._batch_size}"
            )

        engine_outputs = [None for _ in range(engine_inputs_numpy.shape[1])]

        num_batches = engine_inputs_numpy.shape[1] // self._batch_size
        progress = (
            tqdm.tqdm(desc="Inferencing Samples", total=num_batches)
            if self._show_progress_bar
            else None
        )

        futures = [
            self._thread_pool.submit(_engine_forward, batch_origin)
            for batch_origin in range(0, engine_inputs_numpy.shape[1], self._batch_size)
        ]

        wait(futures)

        return engine_outputs

    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        """
        Implements extraction_strategy from the intermediate layer and returns its value

        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        embeddings = []
        for engine_output in engine_outputs:
            assert engine_output.shape[0] == self.sequence_length
            assert engine_output.shape[1] == self._model_size
            if self._extraction_strategy == "per_token":
                embedding = engine_output.flatten().tolist()
            if self._extraction_strategy == "reduce_mean":
                # TODO : https://github.com/deepset-ai/haystack/blob/master/haystack/modeling/model/language_model.py
                vecs = self._pool_tokens(
                    sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token # true
                )
            if self._extraction_strategy == "reduce_max":
                # TODO : https://github.com/deepset-ai/haystack/blob/master/haystack/modeling/model/language_model.py
                vecs = self._pool_tokens(
                sequence_output, padding_mask, self.extraction_strategy, ignore_first_token=ignore_first_token # true
            )
            if self._extraction_strategy == "cls_token":
                embedding = engine_output[0].flatten().tolist()
            else:
                raise ValueError(
                    f"Unsupported extraction_strategy {self._extraction_strategy}"
                )
            embeddings.append(embedding)

        return self.output_schema(embeddings=embeddings)
