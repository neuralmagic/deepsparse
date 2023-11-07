
# DeepSparseSentenceTransformer

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sfN8zDK7MIyatiSIbt2xWh0i6GnaBnTR?usp=sharing)

```python
from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
```

[DeepSparse](https://github.com/neuralmagic/deepsparse) enhances [SentenceTransformers](https://www.sbert.net/), enabling more efficient computation of embeddings for text and images across numerous languages. This improvement hinges on advanced sparse inference methods from DeepSparse and provides performance improvements on CPUs as a result. The system, originally built on PyTorch and Transformers, gains additional muscle from DeepSparse, expanding its repertoire of pre-trained models. It's especially adept at tasks like identifying similar meanings in text, supporting applications in semantic search, paraphrase detection, and more.

## Installation

You can install the DeepSparse SentenceTransformers extension using pip:

```bash
pip install -U deepsparse-nightly[sentence_transformers]
```

## Usage

Using DeepSparse SentenceTransformers is straightforward and similar to the original:

```python
from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
model = DeepSparseSentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', export=True)

# Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding.shape)
    print("")
```

## Benchmarking Performance

There is a `benchmark_encoding.py` script located in this directory that compares a standard model running in both SentenceTransformers and DeepSparse, with a sparsified model in DeepSparse. Make sure to install SentenceTransformers as well if you want to perform this comparison yourself: `pip install sentence_transformers`. Here is an example run on an 4 core SPR CPU with the base model being `BAAI/bge-small-en-v1.5`:
```bash
python benchmark_encoding.py --base_model BAAI/bge-small-en-v1.5 --sparse_model zeroshot/bge-small-en-v1.5-quant

[SentenceTransformer]
Batch size: 1, Sentence length: 700
Latency: 100 sentences in 10.34 seconds
Throughput: 9.67 sentences/second

[DeepSparse Optimized]
Batch size: 1, Sentence length: 700
Latency: 100 sentences in 3.75 seconds
Throughput: 26.65 sentences/second
```


## Accuracy Validation with MTEB

DeepSparse's efficiency doesn't compromise its accuracy, thanks to testing with the Multilingual Text Embedding Benchmark (MTEB). This process validates the model's performance against standard tasks, ensuring its reliability.

To initiate this, you'll need to install MTEB, along with the necessary DeepSparse and SentenceTransformers libraries. Use the following command:

```
pip install mteb deepsparse-nightly[sentence_transformers] sentence-transformers
```

Once installed, you can leverage MTEB for an evaluation as shown in the Python script below:

```python
from mteb import MTEB

# Specify the model to use
model_name = "TaylorAI/bge-micro-v2"

# DeepSparse Model Evaluation
from deepsparse.sentence_transformers import DeepSparseSentenceTransformer
model = DeepSparseSentenceTransformer(model_name, export=True)
evaluation = MTEB(tasks=["Banking77Classification"])
results_ds = evaluation.run(model, output_folder=f"results/ds-{model_name}")
print(results_ds)

# Original SentenceTransformers Model Evaluation
import sentence_transformers
model = sentence_transformers.SentenceTransformer(model_name)
evaluation = MTEB(tasks=["Banking77Classification"])
results_st = evaluation.run(model, output_folder=f"results/st-{model_name}")
print(results_st)
```

Output:
```
{'Banking77Classification': {'mteb_version': '1.1.1', 'dataset_revision': '0fd18e25b25c072e09e0d92ab615fda904d66300', 'mteb_dataset_name': 'Banking77Classification', 'test': {'accuracy': 0.8117207792207791, 'f1': 0.8109893836310513, 'accuracy_stderr': 0.007164150669501205, 'f1_stderr': 0.007346045502756079, 'main_score': 0.8117207792207791, 'evaluation_time': 8.05}}}
{'Banking77Classification': {'mteb_version': '1.1.1', 'dataset_revision': '0fd18e25b25c072e09e0d92ab615fda904d66300', 'mteb_dataset_name': 'Banking77Classification', 'test': {'accuracy': 0.8117207792207791, 'f1': 0.8109893836310513, 'accuracy_stderr': 0.007164150669501205, 'f1_stderr': 0.007346045502756079, 'main_score': 0.8117207792207791, 'evaluation_time': 12.21}}}
```

This script performs a comparative analysis between the DeepSparse-optimized model and the original SentenceTransformers model, using MTEB's "Banking77Classification" task as a benchmark. The results are then saved in separate directories for a clear, side-by-side comparison. This thorough evaluation ensures that the enhancements provided by DeepSparse maintain the high standards of accuracy expected from state-of-the-art NLP models.

---

This documentation is based on the original README from [SentenceTransformers](https://www.sbert.net/). It extends the original functionalities with the optimizations provided by [DeepSparse](https://github.com/neuralmagic/deepsparse).

**Note**: The example usage is designed for the DeepSparse-enhanced version of SentenceTransformers. Make sure to follow the specific installation instructions for full compatibility. Performance optimizations with batching and other advanced features will be part of future updates.
