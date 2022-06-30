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

from typing import List, Union, Optional

import numpy

from deepsparse import Pipeline
from deepsparse.log import get_main_logger
from deepsparse.engine import Context

from haystack.document_stores import BaseDocumentStore
from haystack.nodes import EmbeddingRetriever, DensePassageRetriever
from haystack.nodes.retriever._embedding_encoder import _BaseEmbeddingEncoder
from haystack.nodes.retriever.base import BaseRetriever
from haystack.schema import Document
from haystack.modeling.data_handler.processor import TextSimilarityProcessor

__all__ = [
    "DeepSparseEmbeddingRetriever",
    "DeepSparseDensePassageRetriever"
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
        query_model_path: str = "", # TODO: default
        passage_model_path: str = "",
        max_seq_len_query: int = 64,
        max_seq_len_passage: int = 256,
        top_k: int = 10,
        batch_size: int = 16,
        embed_title: bool = True,
        similarity_function: str = "dot_product",
        progress_bar: bool = True,
        scale_score: bool = True,
        context: Optional[Context] = None,
        **pipeline_kwargs
    ):
        super(BaseRetriever).__init__()

        self.document_store = document_store
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.top_k = top_k
        self.scale_score = scale_score
        self.context = context
        self.use_gpu = False
        self.devices = ["cpu"]

        if document_store is None:
            logger.warning(
                "DensePassageRetriever initialized without a document store. "
                "This is fine if you are performing DPR training. "
                "Otherwise, please provide a document store in the constructor."
            )
        elif document_store.similarity != "dot_product":
            logger.warning(
                f"You are using a Dense Passage Retriever model with the {document_store.similarity} function. "
                "We recommend you use dot_product instead. "
                "This can be set when initializing the DocumentStore"
            )

        if self.context is None:
            self.context = Context(num_cores=None, num_streams=2) # arbitrarily choose 2

        pipeline_kwargs["batch_size"] = batch_size
        pipeline_kwargs["show_progress_bar"] = progress_bar
        pipeline_kwargs["context"] = context

        _LOGGER.info("Creating query pipeline")
        self.query_pipeline = Pipeline.create(
            "embedding_extraction",
            query_model_path,
            emb_extraction_layer=None,
            extraction_strategy="per_token",
            **pipeline_kwargs
        )
        _LOGGER.info("Creating passage pipeline")
        self.passage_pipeline = Pipeline.create(
            "embedding_extraction",
            passage_model_path,
            emb_extraction_layer=None,
            extraction_strategy="per_token",
            **pipeline_kwargs
        )
        _LOGGER.info("Query and passage pipelines initialized")

        # initialize model
        """
        self.processor = TextSimilarityProcessor(
            query_tokenizer=self.query_pipeline.tokenizer,
            passage_tokenizer=self.passage_pipeline.tokenizer,
            max_seq_len_passage=max_seq_len_passage,
            max_seq_len_query=max_seq_len_query,
            label_list=["hard_negative", "positive"],
            metric="text_similarity_metric",
            embed_title=embed_title,
            num_hard_negatives=0,
            num_positives=1,
        )
        """
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_passage = max_seq_len_passage
        self.embed_title = embed_title
        self.model = None
        """
        _DeepSparseBiAdaptiveModel(
            query_pipeline=self.query_pipeline,
            passage_pipeline=self.passage_pipeline,
        )
        """

    def _get_predictions(self, dicts):

        # TODO: fix me please
        print(dicts)
        query_inputs = []
        passage_inputs = []
        for sample in dicts:
            assert len(sample.keys())
            if list(sample.keys())[0] == "query":
                query_inputs.append(sample["query"])
            if list(sample.keys())[0] == "passages":
                passage_inputs.extend(sample["passages"])

        print(query_inputs)
        print(passage_inputs)
        query_outputs = self.query_pipeline(query_inputs)
        passage_outputs = self.passage_pipeline(passage_inputs)

        return {"query": query_outputs, "passage": passage_outputs}


    def train(*args, **kwargs):
        raise NotImplementedError()

    def save(*args, **kwargs):
        raise NotImplementedError()

    def load(*args, **kwargs):
        raise NotImplementedError()

class _DeepSparseBiAdaptiveModel():
    def __init__(
        self,
        query_pipeline,
        passage_pipeline,
    ):
        self._query_pipeline = query_pipeline
        self._passage_pipeline = passage_pipeline

    def forward(self, **kwargs):
        print("_DeepSparseBiAdaptiveModel.foward")
        #TODO

        # extract inputs from kwargs["query_input_ids"] and such
        print(kwargs)

        # run pipelines
        self._query_pipeline()
        self._passage_pipeline()

        # [(query_outputs, passage_outputs)], either can be None
        return [(None, None)]

    def eval(self):
        pass

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
