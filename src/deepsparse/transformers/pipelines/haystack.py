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
Pipeline implementation and pydantic models for Haystack pipeline. Supports a
sample of haystack nodes meant to be used DeepSparseEmbeddingRetriever
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Pipeline
from deepsparse.engine import Context, Scheduler
from deepsparse.pipeline import DEEPSPARSE_ENGINE
from deepsparse.transformers.haystack import (
    DeepSparseDensePassageRetriever as DeepSparseDensePassageRetrieverModule,
)
from deepsparse.transformers.haystack import (
    DeepSparseEmbeddingRetriever as DeepSparseEmbeddingRetrieverModule,
)
from haystack.document_stores import (
    ElasticsearchDocumentStore as ElasticsearchDocumentStoreModule,
)
from haystack.document_stores import FAISSDocumentStore as FAISSDocumentStoreModule
from haystack.document_stores import (
    InMemoryDocumentStore as InMemoryDocumentStoreModule,
)
from haystack.nodes import DensePassageRetriever as DensePassageRetrieverModule
from haystack.nodes import EmbeddingRetriever as EmbeddingRetrieverModule
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
    Schema for inputs to Haystack pipelines
    """

    queries: Union[str, List[str]] = Field(
        description="String or list of strings to query documents with"
    )
    params: Dict[Any, Any] = Field(
        description="Dictionary of params to pass to Haystack pipeline", default={}
    )


class HaystackPipelineOutput(BaseModel):
    """
    Schema for outputs to Haystack pipelines
    """

    documents: Union[List[List[Document]], List[Document]] = Field(
        description="List of document results for each input query"
    )
    root_node: Union[str, List[str]] = Field(
        description="Root node of Haystack Pipeline's graph"
    )
    params: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(
        description="Params passed to Haystack pipeline"
    )
    query: Union[List[str], str] = Field(
        description="Query passed to Haystack Pipeline"
    )
    node_id: Union[List[str], str] = Field(
        description="Node id field from Haystack Pipeline output"
    )


class DocumentStoreType(str, Enum):
    """
    Enum containing all supported haystack document stores
    """

    InMemoryDocumentStore = "InMemoryDocumentStore"
    ElasticsearchDocumentStore = "ElasticsearchDocumentStore"
    FAISSDocumentStore = "FAISSDocumentStore"

    @property
    def constructor(self) -> Type[Any]:
        if self.value == DocumentStoreType.InMemoryDocumentStore:
            return InMemoryDocumentStoreModule
        if self.value == DocumentStoreType.ElasticsearchDocumentStore:
            return ElasticsearchDocumentStoreModule
        if self.value == DocumentStoreType.FAISSDocumentStore:
            return FAISSDocumentStoreModule
        else:
            raise ValueError(f"Unknown enum value {self.value}")

    @property
    def default_args(self) -> Dict[str, Any]:
        if self.value == DocumentStoreType.InMemoryDocumentStore:
            return {
                "similarity": "cosine",
            }
        if self.value == DocumentStoreType.ElasticsearchDocumentStore:
            return {
                "similarity": "cosine",
            }
        if self.value == DocumentStoreType.FAISSDocumentStore:
            return {
                "faiss_index_factory_str": "Flat",
                "similarity": "cosine",
            }
        else:
            raise ValueError(f"Unknown enum value {self.value}")


class RetrieverType(str, Enum):
    """
    Enum containing all supported haystack retrievers
    """

    EmbeddingRetriever = "EmbeddingRetriever"
    DeepSparseEmbeddingRetriever = "DeepSparseEmbeddingRetriever"
    DensePassageRetriever = "DensePassageRetriever"
    DeepSparseDensePassageRetriever = "DeepSparseDensePassageRetriever"

    @property
    def constructor(self) -> Type[Any]:
        if self.value == RetrieverType.EmbeddingRetriever:
            return EmbeddingRetrieverModule
        if self.value == RetrieverType.DeepSparseEmbeddingRetriever:
            return DeepSparseEmbeddingRetrieverModule
        if self.value == RetrieverType.DensePassageRetriever:
            return DensePassageRetrieverModule
        if self.value == RetrieverType.DeepSparseDensePassageRetriever:
            return DeepSparseDensePassageRetrieverModule
        else:
            raise ValueError(f"Unknown enum value {self.value}")

    @property
    def default_args(self) -> Dict[str, Any]:
        if self.value == RetrieverType.EmbeddingRetriever:
            return {
                "pooling_strategy": "reduce_mean",
                "batch_size": 64,
            }
        if self.value == RetrieverType.DeepSparseEmbeddingRetriever:
            return {
                "pooling_strategy": "reduce_mean",
                "emb_extraction_layer": -1,
            }
        if self.value == RetrieverType.DensePassageRetriever:
            return {}
        if self.value == RetrieverType.DeepSparseDensePassageRetriever:
            return {"pooling_strategy": "reduce_mean"}
        else:
            raise ValueError(f"Unknown enum value {self.value}")


class PipelineType(str, Enum):
    """
    Enum containing all supported Haystack pipelines
    """

    DocumentSearchPipeline = "DocumentSearchPipeline"

    @property
    def constructor(self) -> Type[Any]:
        if self.value == PipelineType.DocumentSearchPipeline:
            return DocumentSearchPipelineModule
        else:
            raise ValueError(f"Unknown enum value {self.value}")

    @property
    def default_args(self) -> Dict[str, Any]:
        if self.value == PipelineType.DocumentSearchPipeline:
            return {}
        else:
            raise ValueError(f"Unknown enum value {self.value}")


class HaystackPipelineConfig(BaseModel):
    """
    Schema specifying HaystackPipeline config. Allows for specifying which
    haystack nodes to use and what their arguments should be
    """

    document_store: DocumentStoreType = Field(
        description="Name of haystack document store to use. "
        "Default ElasticsearchDocumentStore",
        default=DocumentStoreType.FAISSDocumentStore,
    )
    document_store_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing document_store",
        default=document_store.default.default_args,
    )
    retriever: RetrieverType = Field(
        description="Name of document retriever to use. Default "
        "DeepSparseEmbeddingRetriever (recommended)",
        default=RetrieverType.DeepSparseEmbeddingRetriever,
    )
    retriever_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing retriever",
        default=retriever.default.default_args,
    )
    haystack_pipeline: PipelineType = Field(
        description="Name of Haystack pipeline to use. Default "
        "DocumentSearchPipeline",
        default=PipelineType.DocumentSearchPipeline,
    )
    haystack_pipeline_args: Dict[str, Any] = Field(
        description="Keyword arguments for initializing haystack_pipeline",
        default=haystack_pipeline.default.default_args,
    )


@Pipeline.register(
    task="haystack",
    task_aliases=[],
    default_model_path="zoo:nlp/masked_language_modeling/bert-base/pytorch/"
    "huggingface/bookcorpus_wikitext/3layer_pruned90-none",
)
class HaystackPipeline(Pipeline):
    """
    Neural Magic's pipeline for running Haystack's DocumentSearchPipeline.
    Supports selected Haystack Nodes as well as Haystack nodes integrated
    with the Neural Magic DeepSparse Engine

    example instantiation:
    ```python
    haystack_pipeline = Pipeline.create(
        task="haystack",
        config: {
            "retriever": "DeepSparseEmbeddingRetriever",
            "retriever_args": {
                "model_path": "masked_language_modeling_model_dir/"
                "extraction_strategy": "reduce_mean"
            }
        },
    )
    ```

    writing documents:
    ```
    haystack_pipeline.write_documents(
        {
            "title": "Claude Shannon"
            "content": "Claude Elwood Shannon was an American mathematician, "
            "electrical engineer, and cryptographer known as a father of "
            "information theory."
        },
        {
            "title": "Vincent van Gogh"
            "content": "Van Gogh was born into an upper-middle-class family. "
            "As a child he was serious, quiet and thoughtful. He began drawing "
            "at an early age and as a young man worked as an art dealer."
        },
        {
            "title": "Stevie Wonder"
            "content": "Stevland Hardaway Morris, known professionally as "
            "Stevie Wonder, is an American singer and musician, who is "
            "credited as a pioneer and influence by musicians across a range "
            "of genres that includes rhythm and blues, pop, soul, gospel, "
            "funk and jazz."
        }
    )
    ```

    example queries:
    ```python
    from deepsparse.transformers.haystack import print_pipeline_documents
    pipeline_outputs = haystack_pipeline(
        queries="who invented information theory",
        params={"Retriever": {"top_k": 4}}
    )
    print_pipeline_documents(pipeline_outputs)
    results = haystack_pipeline(
        queries=[
            "famous artists",
            "what is Stevie Wonder's real name?"
        ],
        params={"Retriever": {"top_k": 4}}
    )
    print_pipeline_documents(pipeline_outputs)
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
    :param docs: list of documents to be written to document_store. Can also
        be written after instantiation with write_documents method.
        Default is None
    :param config: dictionary or instance of HaystackPipelineConfig. Used to
        specify Haystack node arguments
    :param retriever_kwargs: keyword arguments to be passed to retriever. If
        the retriever is a deepsparse retriever, then these arguments will also
        be passed to the EmbeddingExtractionPipeline of the retriever
    """

    def __init__(
        self,
        *,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: Optional[int] = None,
        scheduler: Optional[Scheduler] = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        context: Optional[Context] = None,
        sequence_length: int = 128,
        docs: Optional[List[Dict]] = None,
        config: Optional[Union[HaystackPipelineConfig, Dict[str, Any]]] = None,
        **retriever_kwargs,
    ):
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
                f"batch size set to {retriever_kwargs['batch_size']}"
            )

        # pass arguments to retriever (which then passes to extraction pipeline)
        if "max_seq_len" in retriever_kwargs:
            raise ValueError(
                "Specify max_seq_len by passing sequence_length to pipeline constructor"
            )
        retriever_kwargs["engine_type"] = engine_type
        retriever_kwargs["batch_size"] = batch_size
        retriever_kwargs["num_cores"] = num_cores
        retriever_kwargs["scheduler"] = scheduler
        retriever_kwargs["input_shapes"] = input_shapes
        retriever_kwargs["alias"] = alias
        retriever_kwargs["context"] = context
        retriever_kwargs["max_seq_len"] = sequence_length
        self._config = self._parse_config(config)

        self.initialize_pipeline(retriever_kwargs)
        if docs is not None:
            self.write_documents(docs, overwrite=True)

    def _merge_retriever_args(
        self,
        config_retriever_args: Dict[str, Any],
        init_retriever_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merges retriever args given in config with args given at
        HaystackPipeline initialization. Raises errors for conflicts

        :param config_retriever_args: arguments given in config
        :param init_retriever_kwargs: retriever arguments given at
            HaystackPipeline initialization
        :return: merged arguments from both inputs
        """
        init_retriever_kwargs = init_retriever_kwargs.copy()

        # check for conflicting arguments
        for key in init_retriever_kwargs.keys():
            if key in config_retriever_args.keys():
                raise ValueError(
                    f"Found {key} in both HaystackPipeline arguments and config "
                    "retriever_args. Specify only one"
                )

        config_retriever_args.update(init_retriever_kwargs)
        return config_retriever_args

    def initialize_pipeline(self, init_retriever_kwargs: Dict[str, Any]) -> None:
        """
        Instantiate Haystack nodes needed to run pipeline

        :param init_retriever_kwargs: retriever args passed at the initialization
        of this pipeline
        :return: None
        """
        # merge retriever_args
        if self._config.retriever == RetrieverType.DeepSparseEmbeddingRetriever:
            retriever_args = self._merge_retriever_args(
                self._config.retriever_args, init_retriever_kwargs
            )
        else:
            retriever_args = self._config.retriever_args

        # intialize haystack nodes
        self._document_store = self._config.document_store.constructor(
            **self._config.document_store_args
        )
        self._retriever = self._config.retriever.constructor(
            self._document_store, **retriever_args
        )
        self._haystack_pipeline = self._config.haystack_pipeline.constructor(
            self._retriever, **self._config.haystack_pipeline_args
        )

    def write_documents(
        self, docs: List[Union[Dict[Any, Any], Document]], overwrite: bool = True
    ) -> None:
        """
        Write documents to document_store

        :param docs: list of dicts or Documents to write
        :param overwrite: delete previous documents in store before writing
        :return: None
        """
        if overwrite:
            self._document_store.delete_documents()
        self._document_store.write_documents(docs)
        self._document_store.update_embeddings(self._retriever)

    def _parse_config(
        self,
        config: Optional[Union[HaystackPipelineConfig, dict]],
    ) -> BaseModel:
        """
        :param config: instance of config_schema or dictionary of config values
        :return: instance of config_schema
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
        Run Haystack pipeline

        :param args: input args
        :param kwargs: input kwargs
        :return: outputs from Haystack pipeline. If multiple inputs are passed,
            then each field contains a list of values
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

    def process_pipeline_outputs(
        self, results: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> BaseModel:
        """
        :results: list or instance of a dictionary containing outputs from
            Haystack pipeline
        :return: results cast to output_schema. If multiple inputs are passed,
            then each field contains a list of values
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
        :return: pydantic model class that configs passed to this pipeline must
            comply to
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
