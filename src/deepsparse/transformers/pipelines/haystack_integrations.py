from typing import List, Optional, Union, Dict, Type, Tuple, Any

import torch
import numpy
from haystack.document_stores import BaseDocumentStore
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes import EmbeddingRetriever
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.schema import Document

from deepsparse import Pipeline
from deepsparse.log import get_main_logger

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

        self.embedding_encoder = _DeepSparseEmbeddingEncoder(self, kwargs)

class _DeepSparseEmbeddingEncoder(_BaseEmbeddingEncoder):
    def __init__(self, retriever: "DeepSparseEmbeddingRetriever", kwargs):
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
        document_store = retriever.document_store
        if document_store.similarity != "cosine":
            _LOGGER.warning(
                f"You are using document store embeddings with the "
                f"{document_store.similarity} function. We recommend using "
                "cosine instead. This can be set when initializing DocumentStore"
            )

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[numpy.ndarray]:
        model_output = self.embedding_pipeline(texts)
        embeddings = [numpy.array(embedding) for embedding in model_output.embeddings]
        return embeddings

    def embed_queries(self, texts: List[str]) -> List[numpy.ndarray]:
        return self.embed(texts)

    def embed_documents(self, docs: List[Document]) -> List[numpy.ndarray]:
        passages = [d.content for d in docs]  # type: ignore
        return self.embed(passages)
