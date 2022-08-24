# Haystack: Information Retrieval #
The relevant features added as a part of the Haystack information retrieval integration are a [Haystack pipeline](/src/deepsparse/transformers/haystack/pipeline.py), an [embedding extraction pipeline](/src/deepsparse/transformers/pipelines/embedding_extraction.py), and two classes, [DeepSparseEmbeddingRetriever](/src/deepsparse/transformers/haystack/nodes.py) and [DeepSparseDensePassageRetriever](/src/deepsparse/transformers/haystack/nodes.py).

These features allow a user to perform information retrieval tasks using the Haystack library as well as substitute in sparse retrieval nodes into their existing Haystack systems.

## Installation and Setup ##
In order to ensure the proper installation, please use only install with python version 3.8.

Install `farm-haystack`'s dependencies via deepsparse extras
```bash
pip install deepsparse[haystack]
```

After this is done, importing assets from `deepsparse.transformers.haystack` will trigger an auto-installation of Neural Magic's fork of `transformers` as well as `farm-haystack[all]==1.4.0`. These auto-installations can be controlled by setting the environment variables `NM_NO_AUTOINSTALL_TRANSFORMERS` and `NM_NO_AUTOINSTALL_HAYSTACK` respectively.

## Haystack ##
[Haystack](https://haystack.deepset.ai/overview/intro) is an open source framework developed by Deepset for building document search systems. The library implements classes that handle operations such as document storage, index search, embedding generation, and document search (formally known as information retrieval).

### Document Retrieval with Haystack ###
A typical a document retrieval script in Haystack might look something like this:

First initialize a document store. The document store is responsible for handling the storage of document texts, their embeddings, as well as indexing those embeddings. The simplest document store provided by Haystack is the `InMemoryDocumentStore`, but more complex document stores such as `ElasticDocumentStore`, `FAISSDocumentStore`, or `WeaviateDocumentStore` may require more set up but provide more robust indexing capabilities.
``` python3
from haystack.document_stores import InMemoryDocumentStore

document_store = InMemoryDocumentStore(
    similarity="cosine",
    embedding_dim=768,
    use_gpu=False
)
```

Next, create a retriever. The retriever is houses the deep model and is responsible for, given a document or query, generating an embedding such that query embeddings have a high similarity to their relevant document embeddings.
``` python3
from haystack.nodes import EmbeddingRetriever

retriever = EmbeddingRetriever(
    document_store, # pass in document store
    embedding_model="deepset/sentence_bert",
    use_gpu=False,
)
```
``` python3
>>> retriever.embed_queries(["How many protons in a hydrogen atom"])[0][:10]
array([-0.00331814, -0.16311326, -0.64788855, -0.35724441, -0.26155273,
       -0.76656055,  0.35976224, -0.6578757 , -0.15693564, -0.1927543 ])
```

Next, write some files to your document store. These documents can be instances of Haystack's `Document` class or dictionaries containing a `content`. Remember to update the documents' embeddings with `document_store.update_embeddings(retriever)`
``` python3
document_store.write_documents([
    {
        "title"  : "Looking Glass",
        "content": "He came on a summer's day "
                   "Bringin' gifts from far away."
                   "But he made it clear he couldn't stay."
                   "No harbor was his home."
    },
    {
        "title"  : "Bobby Darin",
        "content": "Somewhere beyond the sea."
                   "Somewhere waiting for me."
                   "My lover stands on golden sands "
                   "And watches the ships that go sailin'"
    }
])
document_store.update_embeddings(retriever)
```

Finally, create a pipeline and run a query using Haystack's `DocumentSearchPipeline`.
``` python3
from haystack.pipelines import DocumentSearchPipeline

pipeline = DocumentSearchPipeline(retriever)
results = pipeline.run(query="Where does my lover stand?", params={"Retriever": {"top_k": 1}})
print(results)
```
```
{'documents': [<Document: {'content': "Somewhere beyond the sea.Somewhere waiting for me.My lover stands on golden sands And watches the ships that go sailin'", 'content_type': 'text', 'score': 0.6692139119642437, 'meta': {'title': 'Bobby Darin'}, 'embedding': None, 'id': '4714818c608d92edccf5ec8b44fde052'}>], 'root_node': 'Query', 'params': {'Retriever': {'top_k': 1}}, 'query': 'Where does my lover stand?', 'node_id': 'Retriever'}
```

### Document Retrieval with DeepSparse ###
To integrate with the DeepSparse Engine, simply replace your Haystack retriever node with an instance of a DeepSparse node.
``` python3
from deepsparse.transformers.haystack import DeepSparseEmbeddingRetriever

retriever = DeepSparseEmbeddingRetriever(
    document_store,
    model_path="zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
)
```

## DeepSparse Nodes ##
DeepSparse Nodes are a set of classes that leverage the embedding extraction pipeline to generate document embeddings using the DeepSparse engine. These embeddings can then be used for information retrieval and other haystack tasks.

### DeepSparseEmbeddingRetriever ###
This class implements Haystack's `EmbeddingRetriever` class with DeepSparse inference using the `EmbeddingExtractionPipeline`. The embedding extraction pipeline takes the passed model path, truncates the ONNX to a transformer layer, then uses those model outputs as embeddings. The embedded representation of the document can be then compared to the embedded representation of the query. Query embeddings and document embeddings that have a high dot_product/cosine similiarity are deemed to be relevant by the `DocumentSearchPipeline`
``` python3
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline

from deepsparse.transformers.haystack import DeepSparseEmbeddingRetriever

document_store = InMemoryDocumentStore(similarity="cosine", embedding_dim=768, use_gpu=False)
document_store.write_documents([
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
    }
])

retriever = DeepSparseEmbeddingRetriever(
    document_store,
    "zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
    pooling_strategy="reduce_mean",
)
document_store.update_embeddings(retriever)

pipeline = DocumentSearchPipeline(retriever)
results = pipeline.run(query="Where does my lover stand?", params={"Retriever": {"top_k": 1}})
```

### DeepSparseDensePassageRetriever ###
This class implements Haystack's `DensePassageRetriever` class with DeepSparse inference using two instances of the  `EmbeddingExtractionPipeline` with shared context. This node takes `query_model_path` and `passage_model_path` as arguments and produces document and query embeddings using their respective models.

Dense passage retrieval requires biencoder models to use. For more support, contact support@neuralmagic.com.

``` python3
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import DocumentSearchPipeline

from deepsparse.transformers.haystack import DeepSparseDensePassageRetriever

document_store = InMemoryDocumentStore(similarity="cosine", embedding_dim=768, use_gpu=False)
document_store.write_documents([
    {
        "content": "High and dry, out of the rain."
                   "It's so easy to hurt others when you can't feel pain. "
                   "And don't you know that a love can't grow "
                   "'Cause there's too much to give, 'cause you'd rather live "
                   "For the thrill of it all."
    },
    {
        "content": "Everybody here is out of sight. "
                   "They don't bark and they don't bite. "
                   "They keep things loose, they keep things light. "
                   "Everybody was dancing in the moonlight. "
    }
])
retriever = DeepSparseDensePassageRetriever(
    document_store,
    query_model_path="./query_model",
    passage_model_path="./passage_model",
    pooling_strategy="cls_token",
)
document_store.update_embeddings(retriever)
pipeline = DocumentSearchPipeline(retriever)

results = pipeline.run(query="How is everybody feeling?", params={"Retriever": {"top_k": 1}})
```

## Haystack Pipeline ##
The haystack pipeline is a non traditional pipeline which constructs Haystack nodes which are used for document retrieval or any other Haystack task. Said another way, this pipeline provides an API for constructing a document_store, retriever, and pipeline like the workflow described in [Document Retrieval with Haystack](#Document-Retrieval-with-Haystack)

This pipeline supports all Haystack document stores, nodes, and pipelines as well as the DeepSparse integrated nodes `DeepSparseEmbeddingRetriever` and `DeepSparseDensePassageRetriever`. Users can control which nodes are included via the `config` argument.
``` python3
from deepsparse import Pipeline
from deepsparse.transformers.haystack import print_pipeline_documents
from deepsparse.transformers.haystack import HaystackPipeline

from haystack.utils import print_documents, fetch_archive_from_http, convert_files_to_docs, clean_wiki_text

documents = [
    {"title": "Rick Astley",
    "content": "Richard Paul Astley (born 6 February 1966) is an English singer, songwriter and "
    "famous musical artist, who has been active in music for several decades. He gained "
    "worldwide fame in the 1980s, having multiple hits including his signature song "
    "Never Gonna Give You Up, Together Forever and Whenever You Need Somebody, and "
    "returned to music full-time in the 2000s after a 6-year hiatus. Outside his "
    "music career, Astley has occasionally worked as a radio DJ and a podcaster."},

    {"title": "Chinese (Language)",
    "content": "Chinese is a group of languages that form the Sinitic branch of the Sino-Tibetan "
    "languages family, spoken by the ethnic Han Chinese majority and many minority "
    "ethnic groups in Greater China. About 1.3 billion people (or approximately 16% "
    "of the world's population) speak a variety of Chinese as their first language."},

    {"title": "Artificial Neural Network",
    "content": "An ANN is based on a collection of connected units or nodes called artificial "
    "neurons, which loosely model the neurons in a biological brain. Each connection, "
    "like the synapses in a biological brain, can transmit a signal to other neurons. "
    "An artificial neuron receives signals then processes them and can signal neurons "
    "connected to it. The signal at a connection is a real number, and the output of "
    "each neuron is computed by some non-linear function of the sum of its inputs."},

    {"title": "Picasso",
    "content": "Pablo Ruiz Picasso (25 October 1881 – 8 April 1973) was a Spanish painter, "
    "sculptor, printmaker, ceramicist and theatre designer who spent most of his adult "
    "life in France. Regarded as one of the most influential painters of the 20th "
    "century, he is known for co-founding the Cubist movement, the invention of "
    "constructed sculpture, the co-invention of collage, and for the wide "
    "variety of styles that he helped develop and explore"},
]

pipeline = HaystackPipeline(
    model_path="zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
    docs=documents,
    config={
        "document_store": "InMemoryDocumentStore",
        "document_store_args": {
            "embedding_dim": 768,
            "similarity": "cosine",
            "use_gpu": False
        },
        "retriever": "DeepSparseEmbeddingRetriever",
        "retriever_args": {
            "pooling_strategy": "reduce_mean"
        },
        "haystack_pipeline": "DocumentSearchPipeline",
    }
)

results = pipeline(queries="Famous artists", params={"Retriever": {"top_k": 1}})
print_pipeline_documents(results)
```
```
Query: Famous artists

{   'content': 'Pablo Ruiz Picasso (25 October 1881 – 8 April 1973) was a '
               'Spanish painter, sculptor, printmaker, ceramicist and theatre '
               'designer who spent most of his adult life in France. Regarded '
               'as one of the most influential painters of the 20th century, '
               'he is known for co-founding the Cubist movement, the invention '
               'of constructed sculpture, the co-invention of collage, and for '
               'the wide variety of styles that he helped develop and explore',
    'name': None}
```

## Embedding Extraction Pipeline
The embedding extraction pipeline is a transformers pipeline that supports the implementation of [DeepSparse Nodes](#DeepSparse-Nodes) as well as the [Haystack Pipeline](#Haystack-Pipeline). It can also be instantiated directly to grab embeddings from any onnx model.

``` python3
from deepsparse import Pipeline

pipeline = Pipeline.create(
    "embedding_extraction",
    model_path="zoo:nlp/masked_language_modeling/distilbert-none/pytorch/huggingface/wikipedia_bookcorpus/pruned80_quant-none-vnni",
    emb_extraction_layer=-1,
    return_numpy=True,
)

text = "sally sold sea shells by the seashore"

embedding = pipeline(text).embeddings[0]
print(embedding)
```

This pipeline works by grabbing embeddings from an intermediate layer of a passed transformer architecture. This is done with the help of `truncate_transformer_onnx_model`, a function that finds the nodes within the onnx graph that mark the last operation performed by a transformer model layer. The onnx model graph is then truncated that node. The embedding extractor pipeline also implements [pooling methods](https://arxiv.org/abs/1806.09828) which help to reduce the dimensionality of embeddings such as `cls_token`, `reduce_mean` `reduce_max`, and `per_token` (None).

## Accuracy Evaluation ##
The DeepSparse nodes were evaluated using evaluation scripts provided by Tevatron. These results are consistent with those documented in [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906).

|Model (WikipediaNQ)|Accuracy @20|Recall|Accuracy @100|Recall|Accuracy @ 200|Recall|
|-|-|-|-|-|-|-|
|base-none|0.7983379501|100.00%|0.863434903|100.00%|0.8842105263|100.00%|
|base-none-untied|0.7988919668|100.07%|0.8581717452|99.39%|0.8842105263|100.00%|
|pruned90-none|0.7878116343|98.68%|0.8584487535|99.42%|0.8770083102|99.19%|
|pruned90-none-untied|0.7828254848|98.06%|0.8570637119|99.26%|0.8747922438|98.93%|
pruned80-vinni|0.7847645429|98.30%|0.856232687|99.17%|0.8717451524|98.59%|
pruned80-vinni-untied|0.7817174515|97.92%|0.8509695291|98.56%|0.8717|98.59%|

|Model (TriviaNQ) |Accuracy @20|Recall|Accuracy @100|Recall|Accuracy @ 200|Recall|
|-|-|-|-|-|-|-|
|base-none|0.7961637055|100.00%|0.853266154|100.00%|0.8672323875|100.00%|
|base-none-untied|0.7943074339|99.77%|0.8503491558|99.66%|0.8661716609|99.88%|
|pruned90-none|0.7839653496|98.47%|0.8440731901|98.92%|0.8609564218|99.28%|
|pruned90-none-untied|0.782904623|98.33%|0.8435428268|98.86%|0.8594537258|99.10%|
|pruned80-vinni|0.7930699196|99.61%|0.8480509149|99.39%|0.8649341466|99.73%|
|pruned80-vinni-untied|0.7867939539|98.82%|0.8460178556|99.15%|0.8629010872|99.50%|

|MSMARCO Passage|MRR@10|Recall|Accuracy @10|Recall|Recall@20|Recall|Recall@100|Recall|Recall@200|Recall|
|-|-|-|-|-|-|-|-|-|-|-|
|base-none|0.3220429117|100.00%|0.6021489971|100.00%|0.6979942693|100.00%|0.8528653295|100.00%|0.8951289398|100.00%|
|base-none-untied|0.3209568722|99.66%|0.5984240688|99.38%|0.6892550143|98.75%|0.8484240688|99.48%|0.8928366762|99.74%|
|pruned90-none|0.3276589007|101.74%|0.6146131805|102.07%|0.7004297994|100.35%|0.8537249284|100.10%|0.8932664756|99.79%|
|pruned90-none-untied|0.3093550166|96.06%|0.588252149|97.69%|0.6793696275|97.33%|0.8368194842|98.12%|0.8812320917|98.45%|
|pruned80-vinni|0.3251235958|100.96%|0.6068767908|100.79%|0.6962750716|99.75%|0.8449856734|99.08%|0.8856733524|98.94%|
|pruned80-vinni-untied|0.3124041479|97.01%|0.5918338109|98.29%|0.6802292264|97.45%|0.8319484241|97.55%|0.8780802292|98.10%|

## Performance Evaluation ##
Retrievers were also evaluated on their run time. This table compares the run time of generating query embeddings using `DenseEmbeddingRetriever` with Pytorch and `DeepSparseEmbeddingRetriever` with the DeepSparse Engine. Both retrievers were evaluated with the same 12 layer BERT model on the same hardware using CPU.

|Number of Queries|DenseEmbeddingRetriever (sec)|DeepSparseEmbeddingRetriever (sec)|
|-|-|-|
|1|0.027|0.012|
|100|1.37|0.96|
|1,000|12.6|9.5|
