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

from typing import List, Optional, Type, Union

import numpy
from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.log import get_main_logger
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.utils import print_documents


__all__ = [
    "DeepSparseEmbeddingRetriever",
    "DeepSparseDensePassageRetriever",
    "print_pipeline_documents",
]


_LOGGER = get_main_logger()


class DeepSparseEmbeddingRetriever(EmbeddingRetriever):
    """
    Deepsparse implementation of Haystack EmbeddingRetriever
    Utilizes EmbeddingExtractionPipeline to create embeddings

    example integration into haystack pipeline:
    ```python
    document_store = ElasticsearchDocumentStore()
    retriever = DeepSparseEmbeddingRetriever(
        document_store=document_store,
        model_path="masked_language_modeling_model_dir/"
    )
    pipeline = DocumentSearchPipeline(retriever)
    ```

    :param document_store: reference to document store to retrieve from
    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
    :param batch_size: number of documents to encode at once
    :param max_seq_len: longest length of each document sequence. Maximum number
        of tokens for the document text. Longer ones will be cut down
    :param pooling_strategy: strategy for combining embeddings
    :param emb_extraction_layer: number of layer from which the embeddings shall
        be extracted. Default: -1 (very last layer)
    :param top_k: how many documents to return per query
    :param progress_bar: if true displays progress bar during embedding
    :param scale_score: whether to scale the similarity score to the unit interval
        (range of [0,1]). If true (default) similarity scores (e.g. cosine or
        dot_product) which naturally have a different value range will be scaled
        to a range of [0,1], where 1 means extremely relevant. Otherwise raw
        similarity scores (e.g. cosine or dot_product) will be used
    :param embed_meta_fields: concatenate the provided meta fields and text
        passage / table to a text pair that is then used to create the embedding.
        This approach is also used in the TableTextRetriever paper and is likely
        to improve  performance if your titles contain meaningful information for
        retrieval (topic, entities etc.).
    :param kwargs: extra arguments passed to EmbeddingExtractionPipeline
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        model_path: str,
        batch_size: int = 1,
        max_seq_len: int = 512,
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,
        top_k: int = 10,
        progress_bar: bool = True,
        scale_score: bool = True,
        embed_meta_fields: List[str] = [],
        **kwargs,
    ):
        super(BaseRetriever).__init__()

        self.document_store = document_store
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.pooling_strategy = pooling_strategy
        self.emb_extraction_layer = emb_extraction_layer
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.scale_score = scale_score
        self.embed_meta_fields = embed_meta_fields

        _LOGGER.info(f"Init retriever using embeddings of model at {model_path}")

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self, kwargs)

    def train(*args, **kwargs):
        raise NotImplementedError()

    def save(*args, **kwargs):
        raise NotImplementedError()

    def load(*args, **kwargs):
        raise NotImplementedError()


class DeepSparseDensePassageRetriever(DensePassageRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_model_path: str = "",  # TODO: default
        passage_model_path: str = "",
        max_seq_len_query: int = 32,
        max_seq_len_passage: int = 32,
        batch_size: int = 1,
        pooling_strategy: str = "per_token",
        top_k: int = 10,
        embed_title: bool = False,
        progress_bar: bool = True,
        scale_score: bool = True,
        context: Optional[Context] = None,
        **pipeline_kwargs,
    ):
        """
        Deepsparse implementation of Haystack DensePassageRetriever
        Utilizes two instances of EmbeddingExtractionPipeline

        example integration into haystack pipeline:
        ```python
        document_store = ElasticsearchDocumentStore()
        retriever = DeepSparseDensePassageRetriever(
            document_store=document_store,
            query_model_path="query_model_dir/",
            passage_model_path="query_model_dir/"
        )
        pipeline = DocumentSearchPipeline(retriever)
        ```

        :param document_store: reference to document store to retrieve from
        :param query_model_path: sparsezoo stub to a query model or (preferred) a
            directory containing a model.onnx, tokenizer config, and model config
        :param passage_model_path: sparsezoo stub to a passage model or (preferred)
            a directory containing a model.onnx, tokenizer config, and model config
        :param max_seq_len_query: longest length of each query sequence. Maximum
            number of tokens for the document text. Longer ones will be cut down
        :param max_seq_len_passage: longest length of each document sequence.
            Maximum number of tokens for the document text. Longer ones will be
            cut down
        :param batch_size: number of documents and queries to encode at once
        :param pooling_strategy: strategy for combining embeddings
        :param top_k: how many documents to return per query
        :param embed_title: True if titles should be embedded into the passage.
            Default is False
        :param progress_bar: if true displays progress bar during embedding
        :param scale_score: whether to scale the similarity score to the unit interval
            (range of [0,1]). If true (default) similarity scores (e.g. cosine or
            dot_product) which naturally have a different value range will be scaled
            to a range of [0,1], where 1 means extremely relevant. Otherwise raw
            similarity scores (e.g. cosine or dot_product) will be used
        :param context: context shared between query and passage models. If None
            is provided, then a new context with 4 streams will be created
        :param pipeline_kwargs: extra arguments passed to EmbeddingExtractionPipeline
        """
        super(BaseRetriever).__init__()

        self.document_store = document_store
        self.batch_size = batch_size
        self.embed_title = embed_title
        self.progress_bar = progress_bar
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.scale_score = scale_score
        self.context = context
        self.use_gpu = False
        self.devices = ["cpu"]

        if document_store is None:
            raise ValueError(
                "DeepSparseDensePassageRetriever must be initialized with a "
                "document_store"
            )
        elif document_store.similarity != "dot_product":
            _LOGGER.warning(
                "You are using a Dense Passage Retriever model with the "
                f"{document_store.similarity} function. We recommend you use "
                "dot_product instead. This can be set when initializing the "
                "DocumentStore"
            )
        if pooling_strategy != "per_token":
            _LOGGER.warning(
                "You are using a Dense Passage Retriever model with "
                f"{pooling_strategy} pooling_strategy. We recommend you use "
                "per_token instead"
            )
        if embed_title:
            raise ValueError(
                "DeepSparseDensePassageRetriever does not support embedding titles"
            )

        if self.context is None:
            self.context = Context(
                num_cores=None, num_streams=4
            )  # arbitrarily choose 4

        _LOGGER.info("Creating query pipeline")
        self.query_pipeline = Pipeline.create(
            "embedding_extraction",
            query_model_path,
            batch_size=batch_size,
            sequence_length=max_seq_len_query,
            emb_extraction_layer=None,
            extraction_strategy=pooling_strategy,
            show_progress_bar=progress_bar,
            context=context,
            **pipeline_kwargs,
        )
        _LOGGER.info("Creating passage pipeline")
        self.passage_pipeline = Pipeline.create(
            "embedding_extraction",
            passage_model_path,
            batch_size=batch_size,
            sequence_length=max_seq_len_passage,
            emb_extraction_layer=None,
            extraction_strategy=pooling_strategy,
            show_progress_bar=progress_bar,
            context=context,
            **pipeline_kwargs,
        )
        _LOGGER.info("Query and passage pipelines initialized")

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.query_pipeline(texts).embeddings

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passage_inputs = [doc.content for doc in docs]
        return self.passage_pipeline(passage_inputs).embeddings

    def _get_predictions(*args, **kwargs):
        raise NotImplementedError()

    def train(*args, **kwargs):
        raise NotImplementedError()

    def save(*args, **kwargs):
        raise NotImplementedError()

    def load(*args, **kwargs):
        raise NotImplementedError()


class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    """
    TODO
    """

    def __init__(self, retriever: DeepSparseEmbeddingRetriever, kwargs):
        self.embedding_pipeline = Pipeline.create(
            "embedding_extraction",
            model_path=retriever.model_path,
            batch_size=retriever.batch_size,
            sequence_length=retriever.max_seq_len,
            emb_extraction_layer=retriever.emb_extraction_layer,
            extraction_strategy=retriever.pooling_strategy,
            show_progress_bar=retriever.progress_bar,
            **kwargs,
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

    def embed(
        self, texts: Union[List[List[str]], List[str], str]
    ) -> List[numpy.ndarray]:
        model_output = self.embedding_pipeline(texts)
        embeddings = [numpy.array(embedding) for embedding in model_output.embeddings]
        return embeddings

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passages = [d.content for d in docs]  # type: ignore
        return self.embed(passages)


def print_pipeline_documents(
    haystack_pipeline_output: Type[BaseModel],
) -> None:
    """
    Helper function to print documents directly from NM Haystack Pipeline outputs

    :param haystack_pipeline_output: instance of HaystackPipelineOutput schema
    :return: None
    """
    documents = haystack_pipeline_output.documents
    if isinstance(documents, list):
        for i in range(len(documents)):
            results_dict = {
                key: value[i] for key, value in haystack_pipeline_output.dict().items()
            }
            print_documents(results_dict)
    else:
        print_documents(documents.dict())
