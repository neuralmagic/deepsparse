# Hugging Face Transformer Inference Pipelines


Hugging Face Transformer integration allows serving and benchmarking sparsified [Hugging Face transformer](https://github.com/huggingface/transformers) models.  
This integration allows for leveraging the DeepSparse Engine to run the transformer inference with GPU-class performance directly on the CPU.

The DeepSparse Engine is taking advantage of sparsity within neural networks to 
reduce compute required as well as accelerate memory-bound workloads. The Engine is particularly effective when leveraging sparsification
methods such as [pruning](https://neuralmagic.com/blog/pruning-overview/) and [quantization](https://arxiv.org/abs/1609.07061). 
These techniques result in significantly more performant and smaller models with limited to no effect on the baseline metrics. 

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

### Model Format
By default, to deploy the transformer using DeepSparse Engine it is required to supply the model in the ONNX format. 
This grants the Engine the flexibility to serve any model in a framework-agnostic environment. 

Below we describe two possibilities to obtain the required ONNX model.

#### Exporting the onnx file from the contents of a local directory
This pathway is relevant if you intend to deploy a model created using [SparseML] (https://github.com/neuralmagic/sparseml) library. 
For more information refer to the appropriate transformers integration documentation in SparseML.
1. The output of the `SparseMl` training is saved to output directory `/{output_dir}` (e.g. `/trained_model`)
2. Depending on the chosen framework, the model files are saved to `model_path`=`/{output_dir}/{framework_name}` (e.g `/trained_model/pytorch`)
3. It is expected that the valid `model_path` contains following, transformer-specific files:
   - `config.json`  
   - `pytorch_model.bin` 
   - `special_tokens_map.json`  
   - `tokenizer_config.json`  
   - `tokenizer.json`  
   - `trainer_state.json`  
   - `training_args.bin`
   - `vocab.txt`

4. To generate an onnx model, refer to the [script for transformer ONNX export](https://github.com/neuralmagic/sparseml/blob/23bea1713f57363caca92b76cb08f0ea2731b1e6/src/sparseml/transformers/export.py).
Example:
```bash
sparseml.transformers.export_onnx --task question-answering --model_path model_path
```
This creates `model.onnx` file, in the parent directory of your `model_path`(e.g. `/trained_model/model.onnx`)

####  Directly using the SparseZoo stub
Alternatively, you can skip the process of onnx model export by downloading all the required model data directly from Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com/).
Example:
```bash
from sparsezoo import Zoo

# you can lookup an appropriate model stub here: https://sparsezoo.neuralmagic.com/
model_stub = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant_6layers-aggressive_96"
# directly download the model data to your local directory
model = Zoo.download_model_from_stub(model_stub)

# the onnx model file is there, ready for deployment
import os 
os.path.isfile(os.path.join(model.dir_path, "model.onnx"))
>>True
```


## Deployment

### Python API
Python API is the default interface for running the inference with the DeepSparse Engine.

The SparseML installation provides a CLI for sparsifying your models for a specific task; appending the `--help` argument displays a full list of options for training in SparseML:
```bash
sparseml.transformers.token_classification --help
```
Output:
```bash
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model, sparsezoo stub. or model identifier from huggingface.co/models (default: None)
  --distill_teacher DISTILL_TEACHER
                        Teacher model which needs to be a trained NER model (default: None)
  --cache_dir CACHE_DIR
                        Where to store the pretrained data from huggingface.co (default: None)
  --recipe RECIPE       
                        Path to a SparseML sparsification recipe, see https://github.com/neuralmagic/sparseml for more information (default: None)
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the datasets library) (default: None)
  ...
```
As indicated above, `model_path` may be a path to a local model directory, however, in the examples below, we set the `model_path` argument to the model stub of our SparseZoo models. 


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

tc_pipeline = pipeline(
    task="text-classification",
    model_path=model_path)

inference = tc_pipeline("Snorlax loves my Tesla!")

>> [{'label': 'LABEL_1', 'score': 0.9884248375892639}]

inference = tc_pipeline("Snorlax hates pineapple pizza!")

>> [{'label': 'LABEL_0', 'score': 0.9981569051742554}]
```

#### Token Classification Pipeline

[List of the Hugging Face SparseZoo Token Classification Models](
https://sparsezoo.neuralmagic.com/?page=1&domain=nlp&sub_domain=token_classification)

```python
from deepsparse.transformers import pipeline

model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"

tc_pipeline = pipeline(
    task="token-classification",
    model_path=model_path,
)

inference = tc_pipeline("Drive from California to Texas!")

>> [{'entity': 'LABEL_0','word': 'drive', ...}, 
    {'entity': 'LABEL_0','word': 'from', ...}, 
    {'entity': 'LABEL_5','word': 'california', ...}, 
    {'entity': 'LABEL_0','word': 'to', ...}, 
    {'entity': 'LABEL_5','word': 'texas', ...}, 
    {'entity': 'LABEL_0','word': '!', ...}]
```

### DeepSparse Server
As an alternative to Python API, the DeepSparse inference server allows you to serve ONNX models and pipelines in HTTP.
To learn more about the DeeepSparse server, refer to the [appropriate documentation](https://github.com/neuralmagic/deepsparse/tree/main/examples/huggingface-transformers).

#### Spinning Up with DeepSparse Server
Install the server:
```bash
pip install deepsparse[server]
```

Run `deepsparse.server --help` to look up the CLI arguments:
```bash
  Start a DeepSparse inference server for serving the models and pipelines
  given within the config_file or a single model defined by task, model_path,
  and batch_size

  Example config.yaml for serving:

  models:
      - task: question_answering
        model_path: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
        batch_size: 1
        alias: question_answering/dense
      - task: question_answering
        ...

Options:
  --host TEXT                     Bind socket to this host. Use --host 0.0.0.0
                                  to make the application available on your
                                  local network. IPv6 addresses are supported,
                                  for example: --host '::'. Defaults to
                                  0.0.0.0
  --port INTEGER                  Bind to a socket with this port. Defaults to
                                  5543.
  --workers INTEGER               Use multiple worker processes. Defaults to
                                  1.
  --log_level [debug|info|warn|critical|fatal]
                                  Sets the logging level. Defaults to info.
  --config_file TEXT              Configuration file containing info on how to
                                  serve the desired models.
  ...
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
### Benchmarking
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