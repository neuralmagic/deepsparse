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

# Use Cases

There are three interfaces for interacting with DeepSparse:

- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with task-specific pre-processing and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on FastAPI and Uvicorn. It enables you to start a model serving endpoint running DeepSparse with a single CLI.

This directory offers examples using each API in various supported tasks. 

### Supported Tasks

DeepSparse supports the following tasks out of the box:

|          NLP          |            CV             |
|-----------------------|---------------------------|
| [Text Classification `"text-classification"`](nlp/text-classification.md)     | [Image Classification `"image_classification"`](cv/image-classification.md)     |
| [Token Classification `"token-classification"`](nlp/token-classification.md)  | [Object Detection `"yolo"`](cv/object-detection-yolov5.md)    |
| [Sentiment Analysis `"sentiment-analysis"`](nlp/sentiment-analysis.md)        | [Instance Segmentation `"yolact"`](cv/image-segmentation-yolact.md)        |
| [Question Answering `"question-answering"`](nlp/question-answering.md)        |                                                                         |
| [Zero-Shot Text Classification `"zero-shot-text-classification"`](nlp/zero-shot-text-classification.md) |                                               |
| [Embedding Extraction `"transformers_embedding_extraction"`](nlp/transformers-embedding-extraction.md) |                                               |

### Examples

**Pipeline Example** | Sentiment Analysis

Here's an example of how a task is used to create a Pipeline:

```python
from deepsparse import Pipeline

pipeline = Pipeline.create(
  task="sentiment_analysis",
  model_path="zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none")

print(pipeline("I love DeepSparse Pipelines!"))
# labels=['positive'] scores=[0.998009443283081]
```

**Server Example** | Sentiment Analysis

Here's an example of how a task is used to create a Server:

```bash
deepsparse.server \
  --task sentiment_analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Making a request:

```python
import requests

# Uvicorn is running on this port
url = 'http://0.0.0.0:5543/v2/models/sentiment_analysis/infer'

# send the data
obj = {"sequences": "Sending requests to DeepSparse Server is fast and easy!"}
resp = requests.post(url=url, json=obj)

# recieve the post-processed output
print(resp.text)
# >> {"labels":["positive"],"scores":[0.9330279231071472]}
```

### Additional Resources

- [Custom Tasks](../user-guide/deepsparse-pipelines.md#custom-use-case)
- [Pipeline User Guide](../user-guide/deepsparse-pipelines.md)
- [Server User Guide](../user-guide/deepsparse-server.md)
