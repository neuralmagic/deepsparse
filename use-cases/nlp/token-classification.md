# Deploying Token Classification Models with DeepSparse

This page explains how to benchmark and deploy a token classification model with DeepSparse.

There are three interfaces for interacting with DeepSparse:
- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing
and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/). It enables you to start a model serving
endpoint running DeepSparse with a single CLI.

## Installation Requirements
This use case requires the installation of [DeepSparse Server](/get-started/install/deepsparse).

Confirm your machine is compatible with our [hardware requirements](/user-guide/deepsparse-engine/hardware-support).

## Benchmarking

We can use the benchmarking utility to demonstrate the DeepSparse's performance. We ran the numbers below on a 23-core server.

### ONNX Runtime Baseline
As a baseline, let's check out ONNX Runtime's performance on BERT. Make sure you have ORT installed (`pip install onnxruntime`).

```bash
deepsparse.benchmark \
  zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/base-none \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e onnxruntime

> Original Model Path: zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/base-none
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 12.2691
> Latency Mean (ms/batch): 5216.3342
> Latency Median (ms/batch): 5216.3342
> Latency Std (ms/batch): 27.7928
> Iterations: 2
```
ONNX Runtime achieves 12 items/second with batch 64 and sequence length 384.
## DeepSparse Speedup
Now, let's run DeepSparse on an inference-optimized sparse version of BERT. This model has been 80% pruned and quantized, while retaining >99% accuracy of the dense baseline on the [conll2003](https://huggingface.co/datasets/conll2003) dataset.
```bash
deepsparse.benchmark \
  zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni \
  -b 64 -s sync -nstreams 1 -i [64,384] \
  -e deepsparse
> Original Model Path: zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni
> Batch Size: 64
> Scenario: sync
> Throughput (items/sec): 99.7367
> Latency Mean (ms/batch): 641.6757
> Latency Median (ms/batch): 641.0878
> Latency Std (ms/batch): 4.0909
> Iterations: 16
```
DeepSparse achieves 100 items/second, a 8x speed-up over ONNX Runtime!
## DeepSparse Engine
Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended using the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 80% pruned-quantized BERT trained on conll2003 from SparseZoo:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path
import numpy as np

# download onnx from sparsezoo and compile with batchsize 1
sparsezoo_stub = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"
batch_size = 1
bert_engine = Engine(
  model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
  batch_size=batch_size   # defaults to batch size 1
)

# input is raw numpy tensors, output is raw scores for classes
inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
output = bert_engine(inputs)
print(output)
# array([[[ 2.0983224 ,  1.2409506 , -1.7314302 , ..., -0.07210742,
#...
#  -2.0502508 , -2.956191  ]]], dtype=float32)]
```
## DeepSparse Pipelines
Pipeline is the default interface for interacting with DeepSparse.

Like Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. This creates a clean API that allows you to pass raw text and images to DeepSparse and receive the post-processed predictions, making it easy to add DeepSparse to your application.

We will use the `Pipeline.create()` constructor to create an instance of a token classification Pipeline with a 80% pruned-quantized version of BERT trained on conll2003. We can then pass raw text to the `Pipeline` and receive the predictions. All of the pre-processing (such as tokenizing the input) is handled by the `Pipeline`.
```python
from deepsparse import Pipeline
task = "ner"
model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"
pipeline = Pipeline.create(
        task=task,
        model_path=model_path,
    )
output = pipeline("Mary is flying from Nairobi to New York")
print(output)
# predictions=[[TokenClassificationResult(entity='LABEL_1', score=0.9949890971183777, word='mary', start=0, end=4, index=1, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9997545480728149, word='is', start=5, end=7, index=2, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9997464418411255, word='flying', start=8, end=14, index=3, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9997068643569946, word='from', start=15, end=19, index=4, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.9992954730987549, word='nairobi', start=20, end=27, index=5, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9997722506523132, word='to', start=28, end=30, index=6, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.9994122385978699, word='new', start=31, end=34, index=7, is_grouped=False), TokenClassificationResult(entity='LABEL_6', score=0.9990378022193909, word='york', start=35, end=39, index=8, is_grouped=False)]]
```
### Use Case Specific Arguments
The Token Classification Pipeline contains additional arguments for configuring a `Pipeline`.

#### Sequence Length
The `sequence_length` argument adjusts the ONNX graph to handle a specific sequence length. In the DeepSparse Pipelines, the tokenizers pad the input. As such, using shorter sequence lengths will have better performance.

The example below compiles the model and runs inference with sequence length of 64.
```python
from deepsparse import Pipeline
task = "ner"
sequence_length = 64
model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"

pipeline = Pipeline.create(
        task=task,
        model_path=model_path,
        sequence_length = sequence_length,
    )
output = pipeline("Mary is flying from Nairobi to New York to attend a conference on generative AI")
print(output)
# predictions=[[TokenClassificationResult(entity='LABEL_1', score=0.9950078129768372, word='mary', start=0, end=4, index=1, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999826192855835, word='is', start=5, end=7, index=2, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998066425323486, word='flying', start=8, end=14, index=3, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998202323913574, word='from', start=15, end=19, index=4, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.9993807077407837, word='nairobi', start=20, end=27, index=5, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999809205532074, word='to', start=28, end=30, index=6, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.999479353427887, word='new', start=31, end=34, index=7, is_grouped=False), TokenClassificationResult(entity='LABEL_6', score=0.9990516901016235, word='york', start=35, end=39, index=8, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998992085456848, word='to', start=40, end=42, index=9, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999877393245697, word='attend', start=43, end=49, index=10, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998823404312134, word='a', start=50, end=51, index=11, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999748945236206, word='conference', start=52, end=62, index=12, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999583899974823, word='on', start=63, end=65, index=13, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.5017374753952026, word='genera', start=66, end=72, index=14, is_grouped=False), TokenClassificationResult(entity='LABEL_8', score=0.892431378364563, word='##tive', start=72, end=76, index=15, is_grouped=False), TokenClassificationResult(entity='LABEL_8', score=0.9190302491188049, word='ai', start=77, end=79, index=16, is_grouped=False)]]
```
Alternatively, you can pass a list of sequence lengths, creating a "bucketable" pipeline. Under the hood, the DeepSparse Pipeline will compile multiple versions of the engine (utilizing a shared scheduler) and direct inputs towards the smallest bucket into which it fits.

The example below creates a bucket for smaller input lengths (16 tokens) and for larger input lengths (128 tokens).
```python
from deepsparse import Pipeline
task = "ner"
buckets = [16, 128]
model_path = "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni"

pipeline = Pipeline.create(
        task=task,
        model_path=model_path,
        sequence_length = buckets,
    )
output = pipeline("Mary is flying from Nairobi to New York to attend a conference on generative AI")
print(output)
# predictions=[[TokenClassificationResult(entity='LABEL_1', score=0.9950078129768372, word='mary', start=0, end=4, index=1, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999826192855835, word='is', start=5, end=7, index=2, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998066425323486, word='flying', start=8, end=14, index=3, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998202323913574, word='from', start=15, end=19, index=4, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.9993807077407837, word='nairobi', start=20, end=27, index=5, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999809205532074, word='to', start=28, end=30, index=6, is_grouped=False), TokenClassificationResult(entity='LABEL_5', score=0.999479353427887, word='new', start=31, end=34, index=7, is_grouped=False), TokenClassificationResult(entity='LABEL_6', score=0.9990516901016235, word='york', start=35, end=39, index=8, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998992085456848, word='to', start=40, end=42, index=9, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999877393245697, word='attend', start=43, end=49, index=10, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.9998823404312134, word='a', start=50, end=51, index=11, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999748945236206, word='conference', start=52, end=62, index=12, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.999583899974823, word='on', start=63, end=65, index=13, is_grouped=False), TokenClassificationResult(entity='LABEL_0', score=0.5017374753952026, word='genera', start=66, end=72, index=14, is_grouped=False), TokenClassificationResult(entity='LABEL_8', score=0.892431378364563, word='##tive', start=72, end=76, index=15, is_grouped=False), TokenClassificationResult(entity='LABEL_8', score=0.9190302491188049, word='ai', start=77, end=79, index=16, is_grouped=False)]]
```
### Cross Use Case Functionality
Check out the [Pipeline User Guide](/user-guide/deepsparse/deepsparse-pipelines) for more details on configuring a Pipeline.
## DeepSparse Server
DeepSparse Server is built on top of FastAPI and Uvicorn, enabling you to set up a REST endpoint for serving inferences over HTTP. Since DeepSparse Server wraps the Pipeline API, it inherits all the utilities provided by Pipelines.

The CLI command below launches a token classification pipeline with a 80% pruned-quantized BERT model:
```bash 
deepsparse.server
--task ner
--model_path "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni" # or path/to/onnx
```
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a /docs path is created with full endpoint descriptions and support for making sample requests.

Here is an example client request, using the Python requests library for formatting the HTTP:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'
# send the data
obj = {
  "inputs": "Mary is flying from Nairobi to New York to attend a conference on generative AI",
}
resp = requests.post(url=url, json=obj)
# receive the post-processed output
print(resp.text)
# {"predictions":[[{"entity":"LABEL_1","score":0.9950078129768372,"index":1,"word":"mary","start":0,"end":4,"is_grouped":false},{"entity":"LABEL_0","score":0.999826192855835,"index":2,"word":"is","start":5,"end":7,"is_grouped":false},{"entity":"LABEL_0","score":0.9998066425323486,"index":3,"word":"flying","start":8,"end":14,"is_grouped":false},{"entity":"LABEL_0","score":0.9998202323913574,"index":4,"word":"from","start":15,"end":19,"is_grouped":false},{"entity":"LABEL_5","score":0.9993807077407837,"index":5,"word":"nairobi","start":20,"end":27,"is_grouped":false},{"entity":"LABEL_0","score":0.999809205532074,"index":6,"word":"to","start":28,"end":30,"is_grouped":false},{"entity":"LABEL_5","score":0.999479353427887,"index":7,"word":"new","start":31,"end":34,"is_grouped":false},{"entity":"LABEL_6","score":0.9990516901016235,"index":8,"word":"york","start":35,"end":39,"is_grouped":false},{"entity":"LABEL_0","score":0.9998992085456848,"index":9,"word":"to","start":40,"end":42,"is_grouped":false},{"entity":"LABEL_0","score":0.999877393245697,"index":10,"word":"attend","start":43,"end":49,"is_grouped":false},{"entity":"LABEL_0","score":0.9998823404312134,"index":11,"word":"a","start":50,"end":51,"is_grouped":false},{"entity":"LABEL_0","score":0.999748945236206,"index":12,"word":"conference","start":52,"end":62,"is_grouped":false},{"entity":"LABEL_0","score":0.999583899974823,"index":13,"word":"on","start":63,"end":65,"is_grouped":false},{"entity":"LABEL_0","score":0.5017374753952026,"index":14,"word":"genera","start":66,"end":72,"is_grouped":false},{"entity":"LABEL_8","score":0.892431378364563,"index":15,"word":"##tive","start":72,"end":76,"is_grouped":false},{"entity":"LABEL_8","score":0.9190302491188049,"index":16,"word":"ai","start":77,"end":79,"is_grouped":false}]]}
```
#### Use Case Specific Arguments
To use the `sequence_length` argument, create a server configuration file for passing the arguments via `kwargs`.

This configuration file sets sequence length to 64:
```yaml
# ner-config.yaml
endpoints:
  - task: ner
    model: zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/12layer_pruned80_quant-none-vnni
    kwargs:
      sequence_length: 64       # uses sequence length 64
```
Spin up the server:

```bash
deepsparse.server \
  --config-file ner-config.yaml
```
Making a request:
```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/predict'

# send the data
obj = {
  "inputs": "Mary is flying from Nairobi to New York to attend a conference on generative AI",
}

resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# {"predictions":[[{"entity":"LABEL_1","score":0.9950078129768372,"index":1,"word":"mary","start":0,"end":4,"is_grouped":false},{"entity":"LABEL_0","score":0.999826192855835,"index":2,"word":"is","start":5,"end":7,"is_grouped":false},{"entity":"LABEL_0","score":0.9998066425323486,"index":3,"word":"flying","start":8,"end":14,"is_grouped":false},{"entity":"LABEL_0","score":0.9998202323913574,"index":4,"word":"from","start":15,"end":19,"is_grouped":false},{"entity":"LABEL_5","score":0.9993807077407837,"index":5,"word":"nairobi","start":20,"end":27,"is_grouped":false},{"entity":"LABEL_0","score":0.999809205532074,"index":6,"word":"to","start":28,"end":30,"is_grouped":false},{"entity":"LABEL_5","score":0.999479353427887,"index":7,"word":"new","start":31,"end":34,"is_grouped":false},{"entity":"LABEL_6","score":0.9990516901016235,"index":8,"word":"york","start":35,"end":39,"is_grouped":false},{"entity":"LABEL_0","score":0.9998992085456848,"index":9,"word":"to","start":40,"end":42,"is_grouped":false},{"entity":"LABEL_0","score":0.999877393245697,"index":10,"word":"attend","start":43,"end":49,"is_grouped":false},{"entity":"LABEL_0","score":0.9998823404312134,"index":11,"word":"a","start":50,"end":51,"is_grouped":false},{"entity":"LABEL_0","score":0.999748945236206,"index":12,"word":"conference","start":52,"end":62,"is_grouped":false},{"entity":"LABEL_0","score":0.999583899974823,"index":13,"word":"on","start":63,"end":65,"is_grouped":false},{"entity":"LABEL_0","score":0.5017374753952026,"index":14,"word":"genera","start":66,"end":72,"is_grouped":false},{"entity":"LABEL_8","score":0.892431378364563,"index":15,"word":"##tive","start":72,"end":76,"is_grouped":false},{"entity":"LABEL_8","score":0.9190302491188049,"index":16,"word":"ai","start":77,"end":79,"is_grouped":false}]]}
```
### Cross Use Case Functionality

Check out the [Server User Guide](/user-guide/deepsparse/deepsparse-server) for more details on configuring the Server.