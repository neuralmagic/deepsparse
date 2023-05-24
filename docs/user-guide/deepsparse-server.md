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

# DeepSparse Server

DeepSparse Server wraps [Pipelines](deepsparse-pipelines.md) with a REST API, making it easy to stand up a inference 
serving endpoint running DeepSparse.

## Quickstart

DeepSparse Server is launched from the CLI. Just like DeepSparse Pipelines, all we 
have to do is pass a task and a model.

Spin up sentiment analysis endpoint with a 90% pruned-quantized BERT model:
```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none 
```

In this case, we used a SparseZoo stub, which instructs the Server to download the relevant
ONNX file from the SparseZoo. To deploy your own model, pass a path to a `model.onnx` file or a 
folder containing the  `model.onnx` and supporting files (e.g., the Hugging Face `tokenizer.json` and `config.json`).

Let's make a request over HTTP. Since the Server is a wrapper around Pipelines, 
we can send raw data to the endpoint and receive the post-processed predictions:

```python
import requests
url = "http://localhost:5543/predict"
obj = {"sequences": "I love querying DeepSparse over HTTP!"}
print(requests.post(url, json=obj).text)

# >>> {"labels":["positive"],"scores":[0.9909943342208862]}
```

For full usage, run:
```bash
deepsparse.server --help
```

## Supported Use Cases

DeepSparse Server supports all tasks available in Pipelines. [Check out the Use Cases page for more details on task-specific APIs](../use-cases).

## Swagger UI

FastAPI's Swagger UI enables you to view your Server's routes and to make sample requests. Navigate to the `/docs` 
route (e.g., `http://localhost:5543/docs`) to try it out.

<p align="center">
  <img src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/server/img/swagger_ui.png"/>
</p>

## Server Configuration

You can configure DeepSparse Server via YAML files.

### Basic Example

Let us walk through a basic example of deploying via a configuration file.

The following creates an endpoint running a 90% pruned-quantized version of 
BERT trained on the SST2 dataset for the sentiment analysis task.

```yaml
# config.yaml
endpoints:
  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

We can then spin up with the `--config-file` argument:

```bash
deepsparse.server \
  --config-file config.yaml
```

Sending a request:
```python
import requests
url = "http://localhost:5543/predict"
obj = {"sequences": "I love querying DeepSparse launched from a config file!"}
print(requests.post(url, json=obj).text)

# >>> {"labels":["positive"],"scores":[0.9136188626289368]}
```

### Server Level Options

At the server level, there are a few arguments that can be toggled.

#### Physical Resources
`num_cores` specifies the number of cores that DeepSparse runs on. By default,
DeepSparse runs on all available cores.

#### Scheduler
`num_workers` configures DeepSparse's scheduler. 

If `num_workers = 1` (the default), DeepSparse uses its "synchronous" scheduler, which allocates as many resources as possible 
to each request. This format is optimizes per-request latency. By setting `num_workers > 1`, DeepSparse 
utilizes its multi-stream scheduler, which processes multiple requests at the same time. 
In deployment scenarios with low batch sizes and high core counts, using the "multi-stream" scheduler 
can increase throughput by allowing DeepSparse to better saturate the cores.

The following configuration creates a Server with DeepSparse running on two cores, with two input streams,
DeepSparse threads pinned to cores, and PyTorch provided with 2 threads.

```yaml
# server-level-options-config.yaml
num_cores: 2
num_workers: 2

endpoints:
  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

We can also adjust the port by providing the `--port` argument.

Spinning up:
```bash
deepsparse.server \
  --config-file server-level-options-config.yaml \
  --port 5555
```

We can then query the Server with the same pattern, querying on port 5555:
```python
import requests
url = "http://localhost:5555/predict"
obj = {"sequences": "I love querying DeepSparse launched from a config file!"}
print(requests.post(url, json=obj).text)

# >>> {"labels":["positive"],"scores":[0.9136188626289368]}
```

### Multiple Endpoints

To serve multiple models from the same context, we can add an additional endpoint
to the server configuration file.

Here is an example which stands up two sentiment analysis endpoints, one using a
dense unoptimized BERT and one using a 90% pruned-quantized BERT.

```yaml
# multiple-endpoint-config.yaml
endpoints:
  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
    route: /sparse/predict
    name: sparse-sentiment-analysis

  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/base-none
    route: /dense/predict
    name: dense-sentiment-analysis
```

Spinning up:
```bash
deepsparse.server \ 
  --config-file multiple-endpoint-config.yaml
```

Making a request:
```python
import requests

obj = {"sequences": "I love querying the multi-model server!"}

sparse_url = "http://localhost:5543/sparse/predict"
print(f"From the sparse model: {requests.post(sparse_url, json=obj).text}")

dense_url = "http://localhost:5543/dense/predict"
print(f"From the dense model: {requests.post(dense_url, json=obj).text}")

# >>> From the sparse model: {"labels":["positive"],"scores":[0.9942120313644409]}
# >>> From the dense model: {"labels":["positive"],"scores":[0.998753547668457]}
```

### Endpoint Level Configuration

We can also configure the properties of each endpoint, including task-specific
arguments from within the YAML file.

For instance, the following configuration file creates two endpoints.

The first is a text classification endpoint, using a 90% pruned-quantized BERT model trained on
IMDB for document classification (which means the model is tuned to classify long 
sequence lengths). We configure this endpoint with batch size 1 and sequence length
of 512. Since sequence length is a task-specific argument used only in Transformers Pipelines, 
we will pass this in `kwargs` in the YAML file.

The second is a sentiment analysis endpoint. We will use the default
sequence length (128) with batch size 3.

```yaml
# advanced-endpoint-config.yaml

endpoints:
  - task: text-classification
    model: zoo:nlp/document_classification/obert-base/pytorch/huggingface/imdb/pruned90_quant-none
    route: /text-classification/predict
    name: text-classification
    batch_size: 1
    kwargs:
      sequence_length: 512  # uses 512 sequence len (transformers pipeline specific)
      top_k: 2              # returns top 2 scores (text-classification pipeline specific arg)

  - task: sentiment-analysis
    model: zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
    route: /sentiment-analysis/predict
    name: sentiment-analysis
    batch_size: 3
```

Spinning up:
```bash
deepsparse.server \
  --config-file advanced-endpoint-config.yaml
```

Making requests:
```python
import requests

# batch 1
document_obj = {"sequences": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. \
    I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, \
    stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure \
    there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and \
    character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. \
    It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden \
    and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say 'Gene Roddenberry's Earth...' otherwise people \
    would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks \
    really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. \
    Jeeez! Dallas all over again."}

# batch 3
short_obj = {"sequences": [
  "I love how easy it is to configure DeepSparse Server!",
  "It was very challenging to configure my old deep learning inference platform",
  "YAML is the best format for configuring my infrastructure"
]}

document_classification_url = "http://localhost:5543/text-classification/predict"
print(requests.post(document_classification_url, json=document_obj).text)

sentiment_analysis_url = "http://localhost:5543/sentiment-analysis/predict"
print(requests.post(sentiment_analysis_url, json=short_obj).text)

# >>> {"labels":[["0","1"]],"scores":[[0.9994900226593018,0.0005100301350466907]]}
# >>> {"labels":["positive","negative","positive"],"scores":[0.9665533900260925,0.9952980279922485,0.9939143061637878]}
```

Check out the [Use Case](../use-cases) page for detailed documentation on task-specific arguments that can be applied to the Server via `kwargs`.

## Custom Use Cases

The endpoints can also take in a custom task, along with custom preprocess and postprocessing functions

```yaml
# custom-processing-config.yaml

endpoints:
  - task: custom
    model: ~/models/resnet50.onnx
    kwargs:
      processing_file: ~/processing.py
```

Where `model` must be a valid onnx model that exists on the system, and `processing_file` must be a
 valid python file contain pre- and/or post-processing functions, the `preprocess` function must take return a list of `numpy.ndarray`(s) and the `postprocess` function must take in a list of `numpy.ndarray`(s) for example:

(make sure you have torchvision installed for this exact example)

```python
# processing.py

from torchvision import transforms
from PIL import Image
import torch
from typing import List

IMAGENET_RGB_MEANS = [0.485, 0.456, 0.406]
IMAGENET_RGB_STDS = [0.229, 0.224, 0.225]
preprocess_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
])

def preprocess(img_file) -> List["numpy.ndarray"]:
    with open(img_file, "rb") as img_file:
        img = Image.open(img_file)
        img = img.convert("RGB")
    img = preprocess_transforms(img)
    batch = torch.stack([img])
    return [batch.numpy()]

def postprocess(outputs: List["numpy.ndarray"]):
    return outputs
```

Spinning up:

```bash
deepsparse.server \
  --config-file custom-processing-config.yaml
```

Now the custom preprocess and postprocess functions will be used when
requests are made to this server!

## Multi-Stream

Stay tuned for documentation on multi-stream scheduling with DeepSparse!

## Logging

Stay tuned for documentation on DeepSparse Logging!

## Hot Reloading

Stay tuned for documentation on Hot Reloading!
