# Hugging Face Transformer Inference Pipelines


Hugging Face Transformer integration allows serving and benchmarking sparsified [Hugging Face transformer](https://github.com/huggingface/transformers) models.  
This integration allows for leveraging the DeepSparse Engine to run the transformer inference with GPU-class performance directly on the CPU.

This integration currently supports several fundamental NLP tasks:
- **Question Answering** - posing questions about a document.
- **Text Classification** - assigning a label or class to a piece of text (e.g Sentiment Analysis task). 
- **Token Classification** - attributing a label to each token in a sentence (e.g. Named Entity Recognition task).

We are actively working on adding more use cases, stay tuned!

## Getting Started


Before you start your adventure with the DeepSparse Engine, make sure that your machine is 
compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation

```pip install deepsparse```

### ONNX Model Support
By default, to deploy the transformer using DeepSparse Engine it is required to supply the model in the ONNX format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic environment. 

Alternatively, instead of the ONNX model, you can also supply:
- a stub to a transformer model from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
- a path to a directory that contains Hugging Face library files (i.e. tokenizer config, model config, etc.).

## Deployment

### Python API
Python API is the default interface for running the inference with the DeepSparse Engine.. 
#### Spinning Up with DeepSparse Engine Python API

This is the fastest way to kick off your experiments with the DeepSparse Engine. You only need to specify the NLP task!
```python
from deepsparse.transformers import pipeline

qa_pipeline = pipeline("question-answering")
inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")

>> {'score': 0.9945447444915771, 'start': 11, 'end': 18, 'answer': 'Snorlax'}
```

While the example above is great for initial baby steps with the Engine, we highly recommend leveraging the sparsity of the SparseZoo models to enable inference on your CPU with GPU-class performance.

In the examples below, we set the `model_path` argument to the model stub of our SparseZoo models. 

#### Question Answering Pipeline

[List of the Hugging Face SparseZoo Question Answering Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=question_answering)

```python
from deepsparse.transformers import pipeline

model_path="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

qa_pipeline = pipeline(
    task="question-answering",
    model_path=model_path)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")

>> {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}
```

#### Text Classification Pipeline

[List of the Hugging Face SparseZoo Text Classification Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=text_classification)

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/12layer_pruned80_quant-none-vnni"

text_classification = pipeline(
    task="text-classification",
    model_path=model_path)

inference = text_classification("Snorlax loves my Tesla!")

>> [{'label': 'LABEL_1', 'score': 0.9884248375892639}]

inference = text_classification("Snorlax hates pineapple pizza!")

>> [{'label': 'LABEL_0', 'score': 0.9981569051742554}]
```

#### Token Classification Pipeline

[List of the Hugging Face SparseZoo Token Classification Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=token_classification)

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"

token_classification = pipeline(
    task="token-classification",
    model_path=model_path,
)

inference = token_classification("I saw Snorlax in Texas!")

>> [{'entity': 'LABEL_0', 'score': 0.99982464, 'index': 1, 'word': 'i', 'start': 0, 'end': 1}, {'entity': 'LABEL_0', 'score': 0.9998014, 'index': 2, 'word': 'saw', 'start': 2, 'end': 5}, ... ]
```

### DeepSparse Server
As an alternative to Python API, the DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP.
To learn more about the DeeepSparse server, refer to the [appropriate documentation](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers).

#### Spinning Up with DeepSparse Server
Install the server:
```bash
pip install deepsparse[server]
```

Example CLI Command to spin up the server:

```bash
deepsparse.server \
    --task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

Sample request to the server:

```python
import requests

url = "http://localhost:5543/predict" # Server's port default to 5543

obj = {
    "question": "Who is Mark?", 
    "context": "Mark is batman."
}

response = requests.post(url, json=obj)
response.text

>> '{"score":0.9534820914268494,"start":8,"end":14,"answer":"batman"}'
```

The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. Want to find out how fast our sparse Hugging Face ONNX models perform inference? 
You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:

```bash
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni

>> Original Model Path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni
>> Batch Size: 1
>> Scenario: multistream
>> Throughput (items/sec): 76.3484
>> Latency Mean (ms/batch): 157.1049
>> Latency Median (ms/batch): 157.0088
>> Latency Std (ms/batch): 1.4860
>> Iterations: 768
```

To learn more about benchmarking, refer to the appropriate documentation.
Also, check out our [Benchmarking tutorial](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark)!

## Tutorials:
For a deeper dive into using transformers within the Neural Magic ecosystem, refer to the detailed tutorials on our [website](https://neuralmagic.com/):
- [Token Classification: Named Entity Recognition](https://neuralmagic.com/use-cases/sparse-named-entity-recognition/)
- [Text Classification: Multi-Class](https://neuralmagic.com/use-cases/sparse-multi-class-text-classification/)
- [Text Classification: Binary](https://neuralmagic.com/use-cases/sparse-binary-text-classification/)
- [Text Classification: Sentiment Analysis](https://neuralmagic.com/use-cases/sparse-sentiment-analysis/)
- [Question Answering](https://neuralmagic.com/use-cases/sparse-question-answering/)

## Support
For Neural Magic Support, sign up or log in to our [Deep Sparse Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ). Bugs, feature requests, or additional questions can also be posted to our [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues).