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
TODO:
"""
from typing import List, Optional, Union, Dict, Type, Tuple, Any

import torch
from enum import Enum
import numpy
from haystack.document_stores import InMemoryDocumentStore as InMemoryDocumentStoreModule
from haystack.document_stores import ElasticsearchDocumentStore as ElasticsearchDocumentStoreModule
from haystack.schema import Document
from haystack.pipelines import DocumentSearchPipeline as DocumentSearchPipelineModule
from haystack.nodes import EmbeddingRetriever as EmbeddingRetrieverModule

from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline
from deepsparse.transformers.pipelines.haystack_integrations import DeepSparseEmbeddingRetriever as DeepSparseEmbeddingRetrieverModule

class HaystackPipelineInput(BaseModel):
    queries: Union[str, List[str]] = Field(
        description="TODO:"
    )
    params: Dict = Field(
        description="TODO:",
        default={}
    )

class HaystackPipelineOutput(BaseModel):
    documents: Union[List[List[Document]], List[Document]] = Field(
        description="TODO:"
    )
    root_node: Union[str, List[str]] = Field(
        description="TODO:"
    )
    params: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(
        description="TODO:"
    )
    query: Union[List[str], str] = Field(
        description="TODO:"
    )
    node_id: Union[List[str], str] = Field(
        description="TODO:"
    )

class HaystackType():
    """
    TODO:
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


class RetrieverType(HaystackType, Enum):
    """
    Enum containing all supported haystack retrievers
    """

    EmbeddingRetriever = "EmbeddingRetriever"
    DeepSparseEmbeddingRetriever = "DeepSparseEmbeddingRetriever"

    _constructor_dict = {
        "EmbeddingRetriever": EmbeddingRetrieverModule,
        "DeepSparseEmbeddingRetriever": DeepSparseEmbeddingRetrieverModule
    }


class PipelineType(HaystackType, Enum):
    """
    Enum containing all supported haystack pipelines
    """

    DocumentSearchPipeline = "DocumentSearchPipeline"

    _constructor_dict = {
        "DocumentSearchPipeline": DocumentSearchPipelineModule
    }


class HaystackPipelineConfig(BaseModel):
    """
    TODO:
    """
    document_store: DocumentStoreType = Field(
        description="TODO",
        default=DocumentStoreType.ElasticsearchDocumentStore
    )
    document_store_args: dict = Field(
        description="TODO",
        default={}
    )
    retriever: RetrieverType = Field(
        description="TODO",
        default=RetrieverType.DeepSparseEmbeddingRetriever
    )
    retriever_args: dict = Field(
        description="TODO",
        default={}
    )
    haystack_pipeline: PipelineType = Field(
        description="TODO",
        default=PipelineType.DocumentSearchPipeline
    )
    haystack_pipeline_args: dict = Field(
        description="TODO",
        default={}
    )


'''
@Pipeline.register(
    task="embedding_extraction",
    task_aliases=[],
    default_model_path=(
        "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
        "wikipedia_bookcorpus/12layer_pruned80_quant-none-vnni"
    ),
)
'''
class HaystackPipeline(TransformersPipeline):
    def __init__(
        self,
        *,
        docs: Optional[List[Dict]] = None,
        config: Optional[Union[HaystackPipelineConfig, dict]] = None,
        **kwargs,
    ):
        # TODO: Assign necessary members

        if kwargs.get("batch_size") and kwargs["batch_size"] != 1:
            raise ValueError(
                f"{self.__class__.__name__} currently only supports batch size 1, "
                f"batch size set to {kwargs['batch_size']}"
            )

        self.docs = docs
        self._config = self._parse_config(config, kwargs)

        self.initialize_pipeline()

    def merge_retriever_args(self, retriever_args, kwargs):
        kwargs = kwargs.copy()

        # Update kwargs names
        if "sequence_length" in kwargs:
            kwargs["max_seq_len"] = kwargs["sequence_length"]

        # If conflicts, throw
        if "max_seq_len" in kwargs and "max_seq_len" in retriever_args:
            raise ValueError(
                "Found sequence_length in pipeline initialization and max_seq_len in retriever args. Use only one"
            )

        for kwarg in kwargs:
            if kwarg in retriever_args.keys():
                raise ValueError(
                    f"Found {kwarg} in both HaystackPipeline arguments and "
                    "config retriever_args. Use only one"
                )

        retriever_args.update(kwargs)
        return retriever_args


    def initialize_pipeline(self):
        self._document_store = self._config.document_store.construct(
            **self._config.document_store_args
        )
        if self.docs is not None:
            self._document_store.delete_documents()
            self._document_store.write_documents(self.docs)

        self._retriever = self._config.retriever.construct(
            self._document_store,
            **self._config.retriever_args
        )
        # TODO: Adjust embedding_dim
        self._document_store.update_embeddings(self._retriever)

        self._haystack_pipeline = self._config.haystack_pipeline.construct(
            self._retriever,
            **self._config.haystack_pipeline_args
        )

    def _parse_config(
        self,
        config: Optional[Union[HaystackPipelineConfig, dict]],
        kwargs: Dict
    ) -> Type[BaseModel]:
        """
        TODO:
        """
        config = config if config else self.config_schema()

        # cast to config_schema
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

        # merge args
        if config.retriever == RetrieverType.DeepSparseEmbeddingRetriever:
            config.retriever_args = self.merge_retriever_args(config.retriever_args, kwargs)

        return config


    def __call__(self, *args, **kwargs) -> BaseModel:
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
            pipeline_results = [self._haystack_pipeline.run(query=query, params=pipeline_inputs.params) for query in pipeline_inputs.queries]
        else:
            pipeline_results = self._haystack_pipeline.run(query=pipeline_inputs.queries, params=pipeline_inputs.params)

        outputs = self.process_pipeline_outputs(
            pipeline_results
        )

        # validate outputs format
        if not isinstance(outputs, self.output_schema):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of "
                f"{self.output_schema} found output of type {type(pipeline_results)}"
            )

        return outputs

    def process_pipeline_outputs(self, results):
        # zip dictionaries for each output

        if isinstance(results, List):
            outputs = {key: [] for key in results[0].keys()}
            for result in results:
                for key, value in result.items():
                    outputs[key].append(value)
        else:
            outputs = results

        return self.output_schema(**outputs)


    #######

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
