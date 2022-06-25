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
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes import EmbeddingRetriever
from haystack.document_stores import BaseDocumentStore, InMemoryDocumentStore
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.schema import Document
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack.modeling.utils import initialize_device_settings

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
        use_gpu: bool = False,
        batch_size: int = 1,
        max_seq_len: int = 512,
        model_format: str = "farm",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1, # TODO: Utilize this
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        scale_score: bool = True,
        **kwargs,
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
        self.emb_extraction_layer = emb_extraction_layer #  TODO: utilize this
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.scale_score = scale_score

        _LOGGER.info(f"Init retriever using embeddings of model at {model_path}")
        if use_gpu:
            _LOGGER.info(
                "DeepSparseEmbeddingRetriever was initialized with use_gpu=True. "
                "However, the deepsparse engine uses cpu. Engine outputs will be "
                "sent to gpu device after inference"
            )

        if devices is not None:
            self.devices = [torch.device(device) for device in devices]
        else:
            self.devices, _ = initialize_device_settings(use_cuda=use_gpu, multi_gpu=True)

        # TODO: Throw value error if unknown model
        # TODO: Throw warning if using wrong kind of model

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self, kwargs)

class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "DeepSparseEmbeddingRetriever", kwargs):
        # TODO: Check imports work
        self.embedding_pipeline = Pipeline.create(
            "embedding_extraction",
            model_path=retriever.model_path,
            batch_size=retriever.batch_size,
            sequence_length=retriever.max_seq_len,
            show_progress_bar=retriever.progress_bar,
            **kwargs
        )

        self.batch_size = retriever.batch_size
        self.show_progress_bar = retriever.progress_bar
        self.use_gpu = retriever.use_gpu
        self.devices = retriever.devices
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
        print(model_output.embeddings[0].dtype)
        if self.use_gpu:
            embeddings = [torch.tensor(embedding, device=self.devices[0]) for embedding in model_output.embeddings]
        return embeddings

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.content] for d in docs]  # type: ignore
        return self.embed(passages)

class HaystackPipelineInput(BaseModel):
    query: Union[str, List[str]] = Field(
        description="TODO:"
    )
    params: Dict = Field(
        description="TODO:",
        default={}
    )

class HaystackPipelineOutput(BaseModel):
    documents: List[Document] = Field(
        description="TODO:"
    )
    root_node: str = Field(
        description="TODO:"
    )
    params: Dict[str, Any] = Field(
        description="TODO:"
    )
    query: Union[str, List[str]] = Field(
        description="TODO:"
    )
    node_id: str = Field(
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

    InMemoryDocumentStore_ = "InMemoryDocumentStore"

    _constructor_dict = {
        "InMemoryDocumentStore": InMemoryDocumentStore
    }


class RetrieverType(HaystackType, Enum):
    """
    Enum containing all supported haystack retrievers
    """

    DeepSparseEmbeddingRetriever_ = "DeepSparseEmbeddingRetriever"

    _constructor_dict = {
        "DeepSparseEmbeddingRetriever": DeepSparseEmbeddingRetriever
    }


class PipelineType(HaystackType, Enum):
    """
    Enum containing all supported haystack pipelines
    """

    DocumentSearchPipeline_ = "DocumentSearchPipeline"

    _constructor_dict = {
        "DocumentSearchPipeline": DocumentSearchPipeline
    }


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
        default={}
    )
    haystack_pipeline: PipelineType = Field(
        description="TODO",
        default=PipelineType.DocumentSearchPipeline_
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
        batch_size: int = 1,
        docs: Optional[List[str]] = None,
        config: Optional[Union[HaystackPipelineConfig, dict]] = None,
        **kwargs,
    ):
        # TODO: Assign necessary members

        # pass to embedding extraction pipeline
        kwargs.update({"batch_size": batch_size})

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
            self._document_store.write_documents(self.docs)

        self._retriever = self._config.retriever.construct(
            self._document_store,
            **self._config.retriever_args
        )
        self._document_store.update_embeddings(self._retriever, update_existing_embeddings=True)

        # TODO: Adjust embedding_dim
        #print(self._retriever.embedding_encoder.embedding_pipeline.)

        self._haystack_pipeline = self._config.haystack_pipeline.construct(
            self._retriever
            # self._config.haystack_pipeline_args
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
        #retriever_args = merge_retriever_args(config.retriever_args, self.kwargs)
        #config.retriever_args = retriever_args
        retriever_args = self.merge_retriever_args(config.retriever_args, kwargs)
        config.retriever_args = retriever_args
        #retriever_args = merge_retriever_args(config.retriever_args, self.kwargs)
        #config.retriever_args = retriever_args

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
        pipeline_results = self._haystack_pipeline.run(query=pipeline_inputs.query, params=pipeline_inputs.params)

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
        print_documents(results, max_text_len=200)
        return self.output_schema(**results)


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
