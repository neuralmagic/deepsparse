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

from enum import Enum
import numpy
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import BaseDocumentStore, InMemoryDocumentStore
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.schema import Document
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

from pydantic import BaseModel, Field

from deepsparse.log import get_main_logger
from deepsparse import Pipeline
from deepsparse.transformers.pipelines import TransformersPipeline

_LOGGER = get_main_logger()

class DeepSparseEmbeddingRetriever(EmbeddingRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        model_path: str,
        model_version: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        max_seq_len: int = 512,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        scale_score: bool = True,
    ):
        super(BaseRetriever).__init__()

        self.document_store = document_store
        self.model_path = model_path
        self.model_format = model_format
        self.model_version = model_version
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.scale_score = scale_score

        _LOGGER.info(f"Init retriever using embeddings of model at {model_path}")

        # TODO: Throw value error if unknown model
        # TODO: Throw warning if using wrong kind of model

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self)

class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "DeepSparseEmbeddingRetriever"):
        # TODO: Check imports work
        self.embedding_pipeline = Pipeline.create(
            "embedding_extraction",
            model_path=retriever.model_path,
            batch_size=retriever.batch_size,
            sequence_length=retriever.max_seq_len,
            show_progress_bar=retriever.progress_bar,
        )

        self.batch_size = retriever.batch_size
        self.show_progress_bar = retriever.progress_bar
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            _LOGGER.warning(
                f"You are using document store embeddings with the "
                f"{document_store.similarity} function. We recommend using "
                "cosine instead. This can be set when initializing DocumentStore"
            )

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[numpy.ndarray]:
        model_output = self.embedding_pipeline(texts)
        embeddings = [embedding for embedding in model_output.embeddings]
        return embeddings

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in docs]  # type: ignore
        return self.embed(passages)


class DocumentStoreType(Enum):
    """
    Enum containing all supported haystack document stores
    """

    InMemoryDocumentStore_ = "InMemoryDocumentStore"

    _constructor_dict = {
        "InMemoryDocumentStore": InMemoryDocumentStore
    }

    @classmethod
    def to_list(cls):
        return cls._value2member_map_

    @property
    def construct(self):
        return self._constructor_dict.value[self.value]


class RetrieverType(Enum):
    """
    Enum containing all supported haystack retrievers
    """

    DeepSparseEmbeddingRetriever_ = "DeepSparseEmbeddingRetriever"

    _constructor_dict = {
        "DeepSparseEmbeddingRetriever": DeepSparseEmbeddingRetriever
    }

    @classmethod
    def to_list(cls):
        return cls._value2member_map_

    @property
    def construct(self):
        return self._constructor_dict.value[self.value]


class PipelineType(Enum):
    """
    Enum containing all supported haystack pipelines
    """

    DocumentSearchPipeline_ = "DocumentSearchPipeline"

    _constructor_dict = {
        "DocumentSearchPipeline": DocumentSearchPipeline
    }

    @classmethod
    def to_list(cls):
        return cls._value2member_map_

    @property
    def construct(self):
        return self._constructor_dict.value[self.value]


class HaystackPipelineConfig(BaseModel):
    """
    TODO:
    """
    document_store: DocumentStoreType = Field(
        description="TODO",
        default=DocumentStoreType.InMemoryDocumentStore_
    )
    document_store_args: dict = Field(
        description="TODO",
        default={"similarity": "cosine", "use_gpu": False, "embedding_dim": 393216}
    )
    retriever: RetrieverType = Field(
        description="TODO",
        default=RetrieverType.DeepSparseEmbeddingRetriever_
    )
    retriever_args: dict = Field(
        description="TODO",
        default={"use_gpu":False, "embedding_dim":393216}
    )
    haystack_pipeline: PipelineType = Field(
        description="TODO",
        default=PipelineType.DocumentSearchPipeline_
    )
    haystack_pipeline_args: dict = Field(
        description="TODO",
        default={"use_gpu":False, "embedding_dim":393216}
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
        sequence_length: int = 512,
        config: Optional[Union[HaystackPipelineConfig, dict]] = None,
        **kwargs,
    ):
        self.sequence_length = sequence_length

        self._config = self._parse_config(config)
        self.initialize_pipeline()

    def initialize_pipeline(self):
        self._document_store = self._config.document_store.construct(
            **self._config.document_store_args
        )
        document_store.write_documents(docs)

        self._retriever = self._config.retriever.construct(
            self._document_store,
            self.model_path
            # self._config.retriever_args
        )
        document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

        # TODO: Adjust embedding_dim
        #print(self._retriever.embedding_encoder.embedding_pipeline.)

        self._haystack_pipeline = self._config.haystack_pipeline.construct(
            self._retriever
            # self._config.haystack_pipeline_args
        )

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

    @property
    def config_schema(self) -> Type[BaseModel]:
        """
        TODO
        """
        return HaystackPipelineConfig

    def _parse_config(self, config: Optional[Union[HaystackPipelineConfig, dict]]) -> Type[BaseModel]:
        """
        TODO:
        """
        config = config if config else {}

        if isinstance(config, self.config_schema):
            return config

        elif isinstance(config, dict):
            return self.config_schema(**config)

        else:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{self.config_schema} object a dict of keywords used to "
                f"construct one. Found {config} instead"
            )

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
        query, params = self.process_inputs(pipeline_inputs)

        pipeline_results = self._haystack_pipeline.run(query=query, params={"Retriever": {"top_k": 2}})
        return engine_outputs

        pipeline_results = self.process_pipeline_outputs(
            pipeline_results, **postprocess_kwargs
        )

        # validate outputs format
        if not isinstance(pipeline_results, self.output_schema):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of "
                f"{self.output_schema} found output of type {type(pipeline_results)}"
            )

        return pipeline_results

    def process_inputs(pipeline_inputs):
        query = "asdf"
        params = {}
        return query, params

    def process_pipeline_outputs(self, results):
        print_documents(results, max_text_len=200)

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
