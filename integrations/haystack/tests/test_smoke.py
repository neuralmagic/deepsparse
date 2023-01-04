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

import pytest
from deepsparse.transformers.haystack import (
    DeepSparseEmbeddingRetriever,
    DeepSparseReader,
)

from haystack.document_stores import InMemoryDocumentStore  # isort:skip
from haystack.pipelines import (
    DocumentSearchPipeline,
    ExtractiveQAPipeline,
)  # isort:skip


@pytest.mark.smoke
def test_document_search():

    document_store = InMemoryDocumentStore(
        similarity="cosine", embedding_dim=768, use_gpu=False
    )
    document_store.write_documents(
        [
            {
                "content": "He came on a summer's day "
                "Bringin' gifts from far away."
                "But he made it clear he couldn't stay."
                "No harbor was his home."
            },
            {
                "content": "Somewhere beyond the sea."
                "Somewhere waiting for me."
                "My lover stands on golden sands "
                "And watches the ships that go sailin'"
            },
        ]
    )

    retriever = DeepSparseEmbeddingRetriever(
        document_store,
        (
            "zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface"
            "/wikipedia_bookcorpus/pruned80_quant-none-vnni"
        ),
        pooling_strategy="reduce_mean",
    )
    document_store.update_embeddings(retriever)

    pipeline = DocumentSearchPipeline(retriever)
    results = pipeline.run(
        query="Where does my lover stand?", params={"Retriever": {"top_k": 1}}
    )
    print(results)


@pytest.mark.smoke
def test_extractive_qa():

    document_store = InMemoryDocumentStore(
        similarity="cosine", embedding_dim=768, use_gpu=False
    )
    document_store.write_documents(
        [
            {
                "content": "He came on a summer's day "
                "Bringin' gifts from far away."
                "But he made it clear he couldn't stay."
                "No harbor was his home."
            },
            {
                "content": "Somewhere beyond the sea."
                "Somewhere waiting for me."
                "My lover stands on golden sands "
                "And watches the ships that go sailin'"
            },
        ]
    )

    retriever = DeepSparseEmbeddingRetriever(
        document_store,
        (
            "zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface"
            "/wikipedia_bookcorpus/pruned80_quant-none-vnni"
        ),
        pooling_strategy="reduce_mean",
    )
    document_store.update_embeddings(retriever)

    reader = DeepSparseReader(
        model_path="zoo:nlp/question_answering/distilbert-none/pytorch/huggingface/squad/pruned80_quant-none-vnni"
    )

    pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

    results = pipeline.run(
        query="Where does my lover stand?",
        params={"Retriever": {"top_k": 1}, "Reader": {"top_k": 1}},
    )
    print(results)
