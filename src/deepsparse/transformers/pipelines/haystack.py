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
Pipeline implementation and pydantic models for haystack pipeline. Supports a
sample of haystack nodes meant to be used DeepSparseEmbeddingRetriever
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.pipeline import Pipeline, DEEPSPARSE_ENGINE
from deepsparse.engine import Scheduler, Context
from deepsparse.transformers.haystack import (
    DeepSparseEmbeddingRetriever as DeepSparseEmbeddingRetrieverModule,
    DeepSparseDensePassageRetriever as DeepSparseDensePassageRetrieverModule
)
from haystack.document_stores import (
    InMemoryDocumentStore as InMemoryDocumentStoreModule,
    ElasticsearchDocumentStore as ElasticsearchDocumentStoreModule,
)
from haystack.nodes import (
    EmbeddingRetriever as EmbeddingRetrieverModule,
    DensePassageRetriever as DensePassageRetrieverModule,
)
from haystack.pipelines import DocumentSearchPipeline as DocumentSearchPipelineModule
from haystack.schema import Document


__all__ = [
    "HaystackPipelineInput",
    "HaystackPipelineOutput",
    "DocumentStoreType",
    "RetrieverType",
    "HaystackPipelineConfig",
    "HaystackPipeline",
]


class HaystackPipelineInput(BaseModel):
    """
    Schema for inputs to haystack pipelines
    """

    queries: Union[str, List[str]] = Field(description="TODO:")
    params: Dict[Any, Any] = Field(description="TODO:", default={})


class HaystackPipelineOutput(BaseModel):
    """
    Schema for outputs to haystack pipelines
    """

    documents: Union[List[List[Document]], List[Document]] = Field(description="TODO:")
    root_node: Union[str, List[str]] = Field(description="TODO:")
    params: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(description="TODO:")
    query: Union[List[str], str] = Field(description="TODO:")
    node_id: Union[List[str], str] = Field(description="TODO:")


class HaystackType:
    """
    Parent class to Haystack node types
    (DocumentStoreType, RetrieverType, PipelineType
    """

    @classmethod
    def to_list(cls):
        return cls._value2member_map_

    @property
    def construct(self):
        return self._constructor_dict.value[self.value]


class DocumentStoreType(HaystackType, Enum):
    """
    Enum containing all supported haystack document stores
    """

    InMemoryDocumentStore = "InMemoryDocumentStore"
    ElasticsearchDocumentStore = "ElasticsearchDocumentStore"

    _constructor_dict = {
        "InMemoryDocumentStore": InMemoryDocumentStoreModule,
        "ElasticsearchDocumentStore": ElasticsearchDocumentStoreModule,
    }

# TODO: the error message for unknown RetrieverType arg is quite messy
class RetrieverType(HaystackType, Enum):
    """
    Enum containing all supported haystack retrievers
    """

    EmbeddingRetriever = "EmbeddingRetriever"
    DeepSparseEmbeddingRetriever = "DeepSparseEmbeddingRetriever"
    DensePassageRetriever = "DensePassageRetriever"
    DeepSparseDensePassageRetriever = "DeepSparseDensePassageRetriever"

    _constructor_dict = {
        "EmbeddingRetriever": EmbeddingRetrieverModule,
        "DeepSparseEmbeddingRetriever": DeepSparseEmbeddingRetrieverModule,
        "DensePassageRetriever": DensePassageRetrieverModule,
        "DeepSparseDensePassageRetriever": DeepSparseDensePassageRetrieverModule
    }


class PipelineType(HaystackType, Enum):
    """
    Enum containing all supported haystack pipelines
    """

    DocumentSearchPipeline = "DocumentSearchPipeline"

    _constructor_dict = {"DocumentSearchPipeline": DocumentSearchPipelineModule}


class HaystackPipelineConfig(BaseModel):
    """
    Schema specifying HaystackPipeline config. Allows for specifying which
    haystack nodes to use and what their arguments should be
    """

    document_store: DocumentStoreType = Field(
        description="Name of haystack document store to use. "
        "Default ElasticsearchDocumentStore",
        default=DocumentStoreType.ElasticsearchDocumentStore,
    )
    document_store_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing document_store", default={}
    )
    retriever: RetrieverType = Field(
        description="Name of document retriever to use. Default "
        "DeepSparseEmbeddingRetriever (recommended)",
        default=RetrieverType.DeepSparseEmbeddingRetriever,
    )
    retriever_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing retriever", default={}
    )
    haystack_pipeline: PipelineType = Field(
        description="Name of haystack pipeline to use. Default "
        "DocumentSearchPipeline",
        default=PipelineType.DocumentSearchPipeline,
    )
    haystack_pipeline_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing haystack_pipeline", default={}
    )


""" TODO
@Pipeline.register(
    task="embedding_extraction",
    task_aliases=[],
    default_model_path=(
        "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
        "wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni"
    ),
)
"""
class HaystackPipeline(Pipeline):
    def __init__(
        self,
        *,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        context: Optional[Context] = None,
        sequence_length: int = 128,
        docs: Optional[List[Dict]] = None,
        config: Optional[Union[HaystackPipelineConfig, dict]] = None,
        **retriever_kwargs,
    ):
        """
        TODO:
        """
        # transformer pipeline members
        self._sequence_length = sequence_length
        self.config = None
        self.tokenizer = None
        self.onnx_input_names = None
        self._temp_model_directory = None

        # pipeline members
        self._model_path_orig = None
        self._model_path = None
        self._engine_type = engine_type
        self._batch_size = batch_size
        self._alias = alias

        if retriever_kwargs.get("batch_size") and retriever_kwargs["batch_size"] != 1:
            raise ValueError(
                f"{self.__class__.__name__} currently only supports batch size 1, "
                f"batch size set to {kwargs['batch_size']}"
            )

        # pass arguments to retriever (which then passes to extraction pipeline)
        if "model_path" in retriever_kwargs or "embedding_model" in retriever_kwargs or "query_model_path" in retriever_kwargs or "passage_model_path" in retriever_kwargs:
            raise ValueError(
                "Found model path in HaystackPipeline arguments. Specify model path "
                "through config.retriever_args instead"
            )
        retriever_kwargs["engine_type"] = engine_type
        retriever_kwargs["batch_size"] = batch_size
        retriever_kwargs["num_cores"] = num_cores
        retriever_kwargs["scheduler"] = scheduler
        retriever_kwargs["input_shapes"] = input_shapes
        retriever_kwargs["alias"] = alias
        retriever_kwargs["context"] = context
        retriever_kwargs["max_seq_len"] = sequence_length
        self._config = self._parse_config(config, retriever_kwargs)

        self.initialize_pipeline(retriever_kwargs)
        if docs is not None:
            self.write_docs(docs, overwrite=True)

    def merge_retriever_args(self, retriever_args: Dict[str, Any], retriever_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO:
        """
        kwargs = retriever_kwargs.copy()

        # check for conflicting arguments
        for kwarg in kwargs:
            if kwarg in retriever_args.keys():
                raise ValueError(
                    f"Found {kwarg} in both HaystackPipeline arguments and "
                    "config retriever_args. Use only one"
                )

        retriever_args.update(kwargs)
        return retriever_args

    def initialize_pipeline(self, retriever_kwargs: Dict[str, Any]) -> None:
        """
        TODO:
        """
        # merge retriever_args
        if self._config.retriever == RetrieverType.DeepSparseEmbeddingRetriever:
            retriever_args = self.merge_retriever_args(
                self._config.retriever_args, retriever_kwargs
            )
        else:
            retriever_args = self._config.retriever_args

        # intialize haystack nodes
        self._document_store = self._config.document_store.construct(
            **self._config.document_store_args
        )
        self._retriever = self._config.retriever.construct(
            self._document_store, **retriever_args
        )
        self._haystack_pipeline = self._config.haystack_pipeline.construct(
            self._retriever, **self._config.haystack_pipeline_args
        )

    def write_docs(self, docs: List[Dict], overwrite: bool = True):
        """
        TODO:
        """
        if overwrite:
            self._document_store.delete_documents()
        self._document_store.write_documents(docs)
        self._document_store.update_embeddings(self._retriever)


    def _parse_config(
        self, config: Optional[Union[HaystackPipelineConfig, dict]], retriever_kwargs: Dict[str, Any]
    ) -> Type[BaseModel]:
        """
        TODO:
        """
        config = config if config else self.config_schema()

        if isinstance(config, self.config_schema):
            pass

        elif isinstance(config, dict):
            config = self.config_schema(**config)

        else:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{self.config_schema} object a dict of keywords used to "
                f"construct one. Found {config} instead"
            )

        return config

    def __call__(self, *args, **kwargs) -> BaseModel:
        """
        TODO:
        """
        if "engine_inputs" in kwargs:
            raise ValueError(
                "invalid kwarg engine_inputs. engine inputs determined "
                f"by {self.__class__.__qualname__}.parse_inputs"
            )

        # parse inputs into input_schema schema if necessary
        pipeline_inputs = self.parse_inputs(*args, **kwargs)
        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )

        # run pipeline
        if isinstance(pipeline_inputs.queries, List):
            pipeline_results = [
                self._haystack_pipeline.run(query=query, params=pipeline_inputs.params)
                for query in pipeline_inputs.queries
            ]
        else:
            pipeline_results = self._haystack_pipeline.run(
                query=pipeline_inputs.queries, params=pipeline_inputs.params
            )

        outputs = self.process_pipeline_outputs(pipeline_results)

        # validate outputs format
        if not isinstance(outputs, self.output_schema):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of "
                f"{self.output_schema} found output of type {type(pipeline_results)}"
            )

        return outputs

    def process_pipeline_outputs(self, results):
        """
        TODO:
        """
        if isinstance(results, List):
            outputs = {key: [] for key in results[0].keys()}
            for result in results:
                for key, value in result.items():
                    outputs[key].append(value)
        else:
            outputs = results

        return self.output_schema(**outputs)

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return HaystackPipelineInput

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return HaystackPipelineOutput

    @property
    def config_schema(self) -> Type[BaseModel]:
        """
        TODO
        """
        return HaystackPipelineConfig

    def setup_onnx_file_path(self) -> str:
        raise NotImplementedError()

    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> BaseModel:
        raise NotImplementedError()

    def process_inputs(
        self,
        inputs: BaseModel,
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        raise NotImplementedError()
