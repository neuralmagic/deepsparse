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

from typing import List, Optional, Union

import numpy
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import BaseReader, DensePassageRetriever, EmbeddingRetriever
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Answer, Document

from deepsparse import Pipeline
from deepsparse.engine import Context
from deepsparse.log import get_main_logger
from deepsparse.transformers.pipelines.question_answering import (
    QuestionAnsweringOutput,
    QuestionAnsweringPipeline,
)


__all__ = [
    "DeepSparseEmbeddingRetriever",
    "DeepSparseDensePassageRetriever",
    "DeepSparseEmbeddingEncoder",
    "DeepSparseReader",
]


_LOGGER = get_main_logger()


class DeepSparseEmbeddingRetriever(EmbeddingRetriever):
    """
    Deepsparse implementation of Haystack EmbeddingRetriever
    Utilizes TransformersEmbeddingExtractionPipeline to create embeddings

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
    :param batch_size: number of documents to encode at once. Default is 1
    :param max_seq_len: longest length of each document sequence. Maximum number
        of tokens for the document text. Longer ones will be cut down
    :param pooling_strategy: strategy for combining embeddings
    :param emb_extraction_layer: f an int, the transformer layer number from
        which the embeddings will be extracted. If a string, the name of last
        ONNX node in model to draw embeddings from. If None, leave the model
        unchanged. Default is -1 (last transformer layer before prediction head)
    :param top_k: how many documents to return per query
    :param progress_bar: if true displays progress bar during embedding.
        Not supported by DeepSparse retriever nodes. Default is False
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
    :param kwargs: extra arguments passed to TransformersEmbeddingExtractionPipeline
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        model_path: str,
        batch_size: int = 1,
        max_seq_len: int = 512,
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: Union[int, str, None] = -1,
        top_k: int = 10,
        progress_bar: bool = False,
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

        if self.batch_size != 1:
            raise ValueError("DeepSparseEmbeddingRetriever only supports batch_size 1")

        _LOGGER.info(f"Init retriever using embeddings of model at {model_path}")
        if self.progress_bar:
            _LOGGER.warn(
                "DeepSparseEmbeddingRetriever does not support progress bar, set "
                "progress_bar to False"
            )

        self.embedding_encoder = DeepSparseEmbeddingEncoder(self, kwargs)

    def train(*args, **kwargs):
        raise NotImplementedError("DeepSparse Engine does not support training")

    def save(*args, **kwargs):
        raise NotImplementedError("DeepSparse Engine does not support saving to files")

    def load(*args, **kwargs):
        raise NotImplementedError(
            "DeepSparse Engine does not support loading from files"
        )


class DeepSparseDensePassageRetriever(DensePassageRetriever):
    """
    Deepsparse implementation of Haystack DensePassageRetriever
    Utilizes two instances of TransformersEmbeddingExtractionPipeline to
    perform query model and passage model inference

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
        number of tokens for the document text. Longer ones will be cut down.
        Default is 32
    :param max_seq_len_passage: longest length of each document sequence.
        Maximum number of tokens for the document text. Longer ones will be
        cut down. Default is 156
    :param batch_size: number of documents and queries to encode at once.
        Default is 1
    :param emb_extraction_layer: if an int, the transformer layer number from
        which the embeddings will be extracted. If a string, the name of last
        ONNX node in model to draw embeddings from. If None, leave the model
        unchanged. Default is -1 (last transformer layer before prediction head)
    :param pooling_strategy: strategy for combining embeddings. Default is
        "cls_token"
    :param top_k: how many documents to return per query. Default is 10
    :param embed_title: True if titles should be embedded into the passage. Raw
        text input will be the title followed by a space followed by the content.
        Default is False
    :param progress_bar: if true displays progress bar during embedding.
        Not supported by DeepSparse retriever nodes. Default is False
    :param scale_score: whether to scale the similarity score to the unit interval
        (range of [0,1]). If true (default) similarity scores (e.g. cosine or
        dot_product) which naturally have a different value range will be scaled
        to a range of [0,1], where 1 means extremely relevant. Otherwise raw
        similarity scores will be used. Default is True
    :param context: context shared between query and passage models. If None
        is provided, then a new context with 4 streams will be created. Default
        is None
    :param pipeline_kwargs: extra arguments passed to
        `TransformersEmbeddingExtractionPipeline`
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        query_model_path,
        passage_model_path,
        max_seq_len_query: int = 32,
        max_seq_len_passage: int = 156,
        batch_size: int = 1,
        emb_extraction_layer: Union[int, str, None] = -1,
        pooling_strategy: str = "cls_token",
        top_k: int = 10,
        embed_title: bool = False,
        progress_bar: bool = False,
        scale_score: bool = True,
        context: Optional[Context] = None,
        **pipeline_kwargs,
    ):
        super(BaseRetriever).__init__()

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.pooling_strategy = pooling_strategy
        self.top_k = top_k
        self.embed_title = embed_title
        self.scale_score = scale_score
        self.context = context
        self.use_gpu = False
        self.devices = ["cpu"]

        if self.progress_bar:
            _LOGGER.warn(
                "DeepSparseDensePassageRetriever does not support progress bar, set "
                "progress_bar to False"
            )

        if "model_path" in pipeline_kwargs:
            del pipeline_kwargs["model_path"]  # ignore model_path argument
        if "max_seq_len" in pipeline_kwargs:
            del pipeline_kwargs["max_seq_len"]  # ignore max_seq_len argument
        if document_store is None:
            raise ValueError(
                "DeepSparseDensePassageRetriever must be initialized with a "
                "document_store"
            )
        if pooling_strategy != "cls_token":
            _LOGGER.warning(
                "You are using a Dense Passage Retriever model with "
                f"{pooling_strategy} pooling_strategy. We recommend you use "
                "cls_token instead"
            )
        if pooling_strategy == "per_token" and max_seq_len_query != max_seq_len_passage:
            raise ValueError(
                "per_token pooling strategy requires that max_seq_len_query "
                f"({max_seq_len_query}) match max_seq_len_passage "
                f"({max_seq_len_passage})"
            )

        if self.context is None:
            self.context = Context()

        _LOGGER.info("Creating query pipeline")
        self.query_pipeline = Pipeline.create(
            "transformers_embedding_extraction",
            query_model_path,
            batch_size=batch_size,
            sequence_length=max_seq_len_query,
            emb_extraction_layer=emb_extraction_layer,
            extraction_strategy=pooling_strategy,
            context=context,
            return_numpy=True,
            **pipeline_kwargs,
        )
        _LOGGER.info("Creating passage pipeline")
        self.passage_pipeline = Pipeline.create(
            "transformers_embedding_extraction",
            passage_model_path,
            batch_size=batch_size,
            sequence_length=max_seq_len_passage,
            emb_extraction_layer=emb_extraction_layer,
            extraction_strategy=pooling_strategy,
            context=context,
            return_numpy=True,
            **pipeline_kwargs,
        )
        _LOGGER.info("Query and passage pipelines initialized")

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        """
        :param texts: list of query strings to embed
        :return: list of embeddings for each query
        """
        return self.query_pipeline(texts).embeddings

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        """
        :param docs: list of document strings to embed
        :return: list of embeddings for each document
        """
        passage_inputs = [self._document_to_passage_input(doc) for doc in docs]
        return self.passage_pipeline(passage_inputs).embeddings

    def train(*args, **kwargs):
        raise NotImplementedError("DeepSparse Engine does not support model training")

    def save(*args, **kwargs):
        raise NotImplementedError("DeepSparse Engine does not support saving to files")

    def load(*args, **kwargs):
        raise NotImplementedError(
            "DeepSparse Engine does not support loading from files"
        )

    def _document_to_passage_input(self, document: Document) -> str:
        # Preprocesses documents to be used as pipeline inputs
        #
        # :param document: document to turn into raw text input
        # :return: raw text input of document title and content
        if (
            hasattr(document, "meta")
            and document.meta.get("title", None) is not None
            and self.embed_title
        ):
            return f"{document.meta['title']} {document.content}"

        return document.content

    def _get_predictions(*args, **kwargs):
        raise NotImplementedError(
            "This helper function is not used by DeepSparseDensePassageRetriever"
        )


class DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    """
    Deepsparse implementation of Haystack EmbeddingEncoder

    :param retriever: retriever that uses this encoder
    :param pipeline_kwargs: extra arguments passed to
        `TransformersEmbeddingExtractionPipeline`
    """

    def __init__(self, retriever: DeepSparseEmbeddingRetriever, pipeline_kwargs):
        self.embedding_pipeline = Pipeline.create(
            "transformers_embedding_extraction",
            model_path=retriever.model_path,
            batch_size=retriever.batch_size,
            sequence_length=retriever.max_seq_len,
            emb_extraction_layer=retriever.emb_extraction_layer,
            extraction_strategy=retriever.pooling_strategy,
            return_numpy=True,
            **pipeline_kwargs,
        )

        self.batch_size = retriever.batch_size
        self.show_progress_bar = retriever.progress_bar
        document_store = retriever.document_store

        if self.show_progress_bar:
            _LOGGER.warn(
                "DeepSparseEmbeddingEncoder does not support progress bar, set "
                "retriever progress_bar to False"
            )
        if document_store.similarity != "cosine":
            _LOGGER.warning(
                f"You are using document store embeddings with the "
                f"{document_store.similarity} function. We recommend using "
                "cosine instead. This can be set when initializing DocumentStore"
            )

    def embed(
        self, texts: Union[List[List[str]], List[str], str]
    ) -> List[numpy.ndarray]:
        """
        :param texts: list of strings to embed
        :return: list of embeddings for each string
        """
        return self.embedding_pipeline(texts).embeddings

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        """
        :param texts: list of query strings to embed
        :return: list of embeddings for each query
        """
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        """
        :param docs: list of document strings to embed
        :return: list of embeddings for each document
        """
        passages = [d.content for d in docs]
        return self.embed(passages)


class DeepSparseReader(BaseReader):
    def __init__(
        self,
        model_path: str,
        top_k=10,
        top_k_per_candidate=3,
        max_seq_len=256,
        doc_stride=128,
        context_window: Union[str, int] = "passage",
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.context_window = context_window
        self.pipeline = QuestionAnsweringPipeline(
            model_path=model_path,
            doc_stride=doc_stride,
            sequence_length=max_seq_len,
            n_best_size=top_k_per_candidate,
            max_answer_length=64,
            **kwargs,
        )

    def predict(self, query: str, documents: List[Document], top_k):
        answers = []
        for doc in documents:
            out: QuestionAnsweringOutput = self.pipeline(
                context=doc.content, question=query
            )
            if self.context_window == "passage":
                start = doc.content.rfind("\n\n", 0, out.start)
                if start < 0:
                    start = doc.content.rfind("\n", 0, out.start)
                if start < 0:
                    start = out.start

                end = doc.content.find("\n\n", out.end, len(doc.content))
                if end < 0:
                    end = doc.content.find("\n", out.end, len(doc.content))
                if end < 0:
                    end = out.end

            else:
                assert isinstance(self.context_window, int)
                start = max(0, out.start - self.context_window)
                end = min(len(doc.content), out.end + self.context_window)
            assert start >= 0 and end >= 0 and end > start, (start, end)
            context = doc.content[start:end].strip()

            answers.append(
                Answer(
                    answer=out.answer,
                    type="extractive",
                    score=out.score,
                    context=context,
                    document_id=doc.id,
                    meta=doc.meta,
                )
            )

        # sort answers by their `score` and select top-k
        answers = sorted(answers, reverse=True)
        answers = answers[: self.top_k]
        return {"query": query, "answers": answers}

    def predict_batch(self, *args, **kwargs):
        raise NotImplementedError
