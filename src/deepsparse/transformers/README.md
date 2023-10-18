# Hugging Face Transformer Inference Pipelines


DeepSparse allows accelerated inference, serving, and benchmarking of sparsified [Hugging Face Transformer](https://github.com/huggingface/transformers) models.  
This integration allows for leveraging the DeepSparse Engine to run the sparsified transformer inference with GPU-class performance directly on the CPU.

The DeepSparse Engine is taking advantage of sparsity within neural networks to 
reduce compute required as well as accelerate memory-bound workloads. The engine is particularly effective when leveraging sparsification
methods such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061). 
These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics. 

This integration currently supports several fundamental NLP tasks:
- **Text Generation** - given the input prompt, generate an output text sequence (e.g. to fill in incomplete text, summarize or paraphrase a text paragraph)
- **Question Answering** - posing questions about a document
- **Sentiment Analysis** - assigning a sentiment to a piece of text
- **Text Classification** - assigning a label or class to a piece of text (e.g duplicate question pairing)
- **Token Classification** - attributing a label to each token in a sentence (e.g. Named Entity Recognition task)

We are actively working on adding more use cases, stay tuned!

## Getting Started

Before you start your adventure with the DeepSparse Engine, make sure that your machine is 
compatible with our [hardware requirements](https://docs.neuralmagic.com/deepsparse/source/hardware.html).

### Installation

```pip install deepsparse[transformers]```

### Model Format
By default, to deploy the transformer using DeepSparse Engine it is required to supply the model in the ONNX format along with the HuggingFace supporting files. 
This grants the engine the flexibility to serve any model in a framework-agnostic environment. 

The DeepSparse pipelines require the following files within a folder on the local server to properly load a Transformers model:
- `model.onnx`: The exported Transformers model in the [ONNX format](https://github.com/onnx/onnx).
- `config.json`: The [HuggingFace compatible configuration file](https://huggingface.co/docs/transformers/main_classes/configuration) used with the model.
- `tokenizer_config.json`: The [HuggingFace compatible tokenizer configuration](https://huggingface.co/docs/transformers/fast_tokenizers) used with the model.
- `tokenizer.json`, `special_tokens_map.json`, `vocab.json`, `merges.txt` (optional): Other files that may be required by a tokenizer
Below we describe two possibilities to obtain the required structure.

#### SparseML Export 
This pathway is relevant if you intend to deploy a model created using [SparseML](https://github.com/neuralmagic/sparseml) library. 
For more information, refer to the appropriate [transformers integration documentation in SparseML](https://github.com/neuralmagic/sparseml/tree/main/src/sparseml/transformers).

ONNX models can be exported using the `sparseml.transformers.export_onnx` tool:

```bash
sparseml.transformers.export_onnx --task question-answering --model_path model_path
```

This creates `model.onnx` file, in the directory of your `model_path`(e.g. `/trained_model/model.onnx`). 
Any additional, required files, such as e.g.`tokenizer.json` or `config.json`, are stored under the `model_path` folder as well, so a DeepSparse pipeline can be directly instantiated by using that folder after export (e.g. `/trained_model/`).

####  SparseZoo Stub
Alternatively, you can skip the process of the ONNX model export by using Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/). The SparseZoo contains pre-sparsified models and SparseZoo stubs enable you to reference any model on the SparseZoo in a convenient and predictable way.
All of DeepSparse's pipelines and APIs can use a SparseZoo stub in place of a local folder. The Deployment APIs examples use SparseZoo stubs to highlight this pathway.

## Deployment APIs

DeepSparse provides both a Python Pipeline API and an out-of-the-box model server
that can be used for end-to-end inference in either existing python workflows or as an HTTP endpoint.
Both options provide similar specifications for configurations and support a variety of NLP transformers
tasks including question answering, text classification, sentiment analysis, and token classification.

### Python Pipelines
Pipelines are the default interface for running inference with the DeepSparse Engine.

Once a model is obtained, either through `SparseML` training or directly from `SparseZoo`,
`deepsparse.Pipeline` can be used to easily facilitate end to end inference and deployment
of the sparsified transformers model.

If no model is specified to the `Pipeline` for a given task, the `Pipeline` will automatically
select a pruned and quantized model for the task from the `SparseZoo` that can be used for accelerated
inference. Note that other models in the SparseZoo will have different tradeoffs between speed, size,
and accuracy.

### HTTP Server
As an alternative to Python API, the DeepSparse Server allows you to serve ONNX models and pipelines in HTTP.
Both configuring and making requests to the server follow the same parameters and schemas as the
Pipelines enabling simple deployment.  Once launched, a `/docs` endpoint is created with full
endpoint descriptions and support for making sample requests.

Example deployments using NLP transformer models are provided below.
For full documentation on deploying sparse transformer models with the DeepSparse Server, see the
[documentation](https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server).

#### Installation
The DeepSparse Server requirements can be installed by specifying the `server` extra dependency when installing
DeepSparse.

```bash
pip install deepsparse[server,transformers]
```

## Deployment Use Cases
The following section includes example usage of the Pipeline and server APIs for various NLP transformers tasks.

### Question Answering
The question answering tasks accepts a `question` and a `context`. The pipeline will predict an answer
for the `question` as a substring of the `context`.  The following examples use a pruned and quantized
question answering BERT model trained on the `SQuAD` dataset downloaded by default from the SparseZoo.

[List of available SparseZoo Question Answering Models](
https://sparsezoo.neuralmagic.com/?useCase=question_answering)

#### Python Pipeline

```python
from deepsparse import Pipeline

qa_pipeline = Pipeline.create(task="question-answering")
inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")

>> {'score': 0.9947717785835266, 'start': 11, 'end': 18, 'answer': 'Snorlax'}
```

#### HTTP Server
Spinning up:
```bash
deepsparse.server \
    task question-answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"
```

Making a request:
```python
import requests

url = "http://localhost:5543/v2/models/question_answering/infer" # Server's port default to 5543

obj = {
    "question": "Who is Mark?", 
    "context": "Mark is batman."
}

response = requests.post(url, json=obj)
response.text

>> '{"score":0.9534820914268494,"start":8,"end":14,"answer":"batman"}'
```

### Text Generation
The text generation task generates a sequence of tokens given the prompt. Popular text generation LLMs (Large Language Models) are used
for the chatbots (the instruction models), code generation, text summarization, or filling out the missing text. The following example uses a sparsified text classification
OPT model to complete the prompt

[List of available SparseZoo Text Generation Models](
https://sparsezoo.neuralmagic.com/?useCase=text_generation)

#### Python Pipeline
```python
from deepsparse import Pipeline

opt_pipeline = Pipeline.create(task="opt")

inference = opt_pipeline("Who is the president of the United States?")

>> 'The president of the United States is the head of the executive branch of government...'
```

#### HTTP Server
Spinning up:
```bash
deepsparse.server \
    task text-generation \
    --model_path # TODO: Pending until text generation models get uploaded to SparseZoo
```

Making a request:
```python
import requests

url = "http://localhost:5543/v2/models/text_generation/infer" # Server's port default to 5543

obj = {"sequence": "Who is the president of the United States?"}

response = requests.post(url, json=obj)
response.text

>> 'The president of the United States is the head of the executive branch of government...'
```

### Sentiment Analysis
The sentiment analysis task takes in a sentence and classifies its sentiment. The following example
uses a pruned and quantized text sentiment analysis BERT model trained on the `sst2` dataset downloaded
from the SparseZoo. This `sst2` model classifies sentences as positive or negative.

[List of available SparseZoo Sentiment Analysis Models](
https://sparsezoo.neuralmagic.com/?useCase=sentiment_analysis)

#### Python Pipeline
```python
from deepsparse import Pipeline

sa_pipeline = Pipeline.create(task="sentiment-analysis")

inference = sa_pipeline("Snorlax loves my Tesla!")

>> [{'label': 'LABEL_1', 'score': 0.9884248375892639}]  # positive sentiment

inference = sa_pipeline("Snorlax hates pineapple pizza!")

>> [{'label': 'LABEL_0', 'score': 0.9981569051742554}]  # negative sentiment
```

#### HTTP Server
Spinning up:
```bash
deepsparse.server \
    task sentiment-analysis \
    --model_path "zoo:nlp/sentiment_analysis/bert-base/pytorch/huggingface/sst2/pruned80_quant-none-vnni"
```

Making a request:
```python
import requests

url = "http://localhost:5543/v2/models/sentiment_analysis/infer" # Server's port default to 5543

obj = {"sequences": "Snorlax loves my Tesla!"}

response = requests.post(url, json=obj)
response.text

>> '{"labels":["LABEL_1"],"scores":[0.9884248375892639]}'
```

### Text Classification
The text classification task supports binary, multi class, and regression predictions over
sentence inputs. The following example uses a pruned and quantized text classification
DistilBERT model trained on the `qqp` dataset downloaded from a SparseZoo stub.
The `qqp` dataset takes pairs of questions and predicts if they are a duplicate or not.

[List of available SparseZoo Text Classification Models](
https://sparsezoo.neuralmagic.com/?useCase=text_classification)

#### Python Pipeline
```python
from deepsparse import Pipeline

tc_pipeline = Pipeline.create(
   task="text-classification",
   model_path="zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni",
)

# inference of duplicate question pair
inference = tc_pipeline(
   sequences=[
      [
         "Which is the best gaming laptop under 40k?",
         "Which is the best gaming laptop under 40,000 rs?",
      ]
   ]
)

>> TextClassificationOutput(labels=['duplicate'], scores=[0.9947025775909424])
```

#### HTTP Server
Spinning up:
```bash
deepsparse.server \
    task text-classification \
    --model_path "zoo:nlp/text_classification/distilbert-none/pytorch/huggingface/qqp/pruned80_quant-none-vnni"
```

Making a request:
```python
import requests

url = "http://localhost:5543/v2/models/text_classification/infer" # Server's port default to 5543

obj = {
    "sequences": [
        [
            "Which is the best gaming laptop under 40k?",
            "Which is the best gaming laptop under 40,000 rs?",
        ]
    ]
}

response = requests.post(url, json=obj)
response.text

>> '{"labels": ["duplicate"], "scores": [0.9947025775909424]}'
```

### Token Classification Pipeline
The token classification task takes in sequences as inputs and assigns a class to each token.
The following example uses a pruned and quantized token classification NER BERT model
trained on the `CoNLL` dataset downloaded from the SparseZoo.

[List of available SparseZoo Token Classification Models](
https://sparsezoo.neuralmagic.com/?useCase=token_classification)

#### Python Pipeline
```python
from deepsparse import Pipeline

# default model is a pruned + quantized NER model trained on the CoNLL dataset
tc_pipeline = Pipeline.create(task="token-classification")
inference = tc_pipeline("Drive from California to Texas!")

>> [{'entity': 'LABEL_0','word': 'drive', ...}, 
    {'entity': 'LABEL_0','word': 'from', ...}, 
    {'entity': 'LABEL_5','word': 'california', ...}, 
    {'entity': 'LABEL_0','word': 'to', ...}, 
    {'entity': 'LABEL_5','word': 'texas', ...}, 
    {'entity': 'LABEL_0','word': '!', ...}]
```

#### HTTP Server
Spinning up:
```bash
deepsparse.server \
    task token-classification \
    --model_path "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/pruned90-none"
```

Making a request:
```python
import requests

url = "http://localhost:5543/v2/models/token_classification/infer" # Server's port default to 5543

obj = {"inputs": "Drive from California to Texas!"}


response = requests.post(url, json=obj)
response.text

>> '{"predictions":[[{"entity":"LABEL_0","score":0.9998655915260315,"index":1,"word":"drive","start":0,"end":5,"is_grouped":false},{"entity":"LABEL_0","score":0.9998604655265808,"index":2,"word":"from","start":6,"end":10,"is_grouped":false},{"entity":"LABEL_5","score":0.9994636178016663,"index":3,"word":"california","start":11,"end":21,"is_grouped":false},{"entity":"LABEL_0","score":0.999838650226593,"index":4,"word":"to","start":22,"end":24,"is_grouped":false},{"entity":"LABEL_5","score":0.9994573593139648,"index":5,"word":"texas","start":25,"end":30,"is_grouped":false},{"entity":"LABEL_0","score":0.9998716711997986,"index":6,"word":"!","start":30,"end":31,"is_grouped":false}]]}'
```

## Benchmarking
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
