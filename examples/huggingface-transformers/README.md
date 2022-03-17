<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Transformers 🤗 Inference Pipelines
This directory contains examples for serving, benchmarking, and running NLP models from the [Transformers](https://github.com/huggingface/transformers) repository using the DeepSparse Engine. These examples can load pre-trained, sparsified models from SparseZoo or you can specify your own transformer ONNX file. In addition, we also highlight how you can easily perform benchmarking and deploy transformers with the `deepsparse.server` via simple CLI commands.

### Installation

```bash
pip install deepsparse
```

The DeepSparse-Hugging Face pipeline integration provides a simple API dedicated to several tasks:
### Token Classification Pipeline
```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/base-none"

token_classification = pipeline(
    task="token-classification",
    model_path=model_path,
    num_cores=None
)

inference = token_classification("I saw Snorlax in Texas!")
```

### Text Classification | Sentiment Analysis Pipeline

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/text_classification/bert-base/pytorch/huggingface/sst2/base-none"

text_classification = pipeline(
    task="text-classification",
    model_path=model_path,
    num_cores=None
)

inference = text_classification("Snorlax loves my Tesla!")
```

### Question Answering Pipeline

```python
from deepsparse.transformers import pipeline

model_path="zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98"

qa_pipeline = pipeline(
    task="question-answering",
    model_path=model_path,
    num_cores=None,
)

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```
💡**PRO-TIP**💡: The pipeline can also infer a default sparse model to run automatically:

```python
from deepsparse.transformers import pipeline

qa_pipeline = pipeline("question-answering")

inference = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```
**more tasks coming soon...😇**
## Benchmarking & the DeepSparse Server

### Benchmarking 📜

💾 [List of the the Hugging Face SparseZoo Models](https://sparsezoo.neuralmagic.com/?repo=huggingface&page=1)

Want to find out how fast our sparse Hugging Face ONNX models perform inference? You can quickly do benchmarking tests with CLI commands; you only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:

```bash
deepsparse.benchmark zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98
```

For a more in-depth discussion on benchmarking, check out the [Benchmarking tutorial]("https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/benchmark_model")!

### DeepSparse Server 🔌

The DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP. 

##### Single Model Inference

Example CLI command for serving a single model:

```bash
deepsparse.server \
    --task question_answering \
    --model_path "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98"
```

##### Multiple Model Inference
To serve multiple models you can easily build a `config.yaml` file. 
In the sample yaml below, we are defining 2 BERT models to be served by the `deepsparse.server` for the question answering task:

    models:
        - task: question_answering
            model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
            batch_size: 1
            alias: question_answering/dense
        - task: question_answering
            model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98
            batch_size: 1
            alias: question_answering/sparse_quantized

After you finish building the `config.yaml` file, you can run the server with the config file path passed in the `--config_file` argument:
```bash
deepsparse.server --config_file config.yaml
```

For a more in-depth discussion on running the `deepsparse.server`, check out the [DeepSparse Server tutorial]("https://github.com/neuralmagic/deepsparse/tree/main/src/deepsparse/server")!