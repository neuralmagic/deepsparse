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

import torch
from deepsparse import Pipeline
from deepsparse.log import get_main_logger
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document


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
        model_format: str = "neuralmagic",
        pooling_strategy: str = "reduce_mean",
        emb_extraction_layer: int = -1,  # TODO: Utilize this
        top_k: int = 10,
        progress_bar: bool = True,
        devices: Optional[List[Union[str, torch.device]]] = None,
        scale_score: bool = True,
        embed_meta_fields: List[str] = [],
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
        self.emb_extraction_layer = emb_extraction_layer  #  TODO: utilize this
        self.top_k = top_k
        self.progress_bar = progress_bar
        self.scale_score = scale_score
        self.embed_meta_fields = embed_meta_fields

        _LOGGER.info(f"Init retriever using embeddings of model at {model_path}")

        if use_gpu:
            _LOGGER.warn(
                f"DeepSparseEmbeddingRetriever does not use gpu, set use_gpu to False"
            )

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self, kwargs)


class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "DeepSparseEmbeddingRetriever", kwargs):
        self.embedding_pipeline = Pipeline.create(
            "embedding_extraction",
            model_path=retriever.model_path,
            batch_size=retriever.batch_size,
            sequence_length=retriever.max_seq_len,
            emb_extraction_layer=retriever.emb_extraction_layer,
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
