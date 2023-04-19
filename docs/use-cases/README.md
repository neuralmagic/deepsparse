# Use Cases

There are three interfaces for interacting with DeepSparse:

- **Engine** is the lowest-level API that enables you to compile a model and run inference on raw input tensors.

- **Pipeline** is the default DeepSparse API. Similar to Hugging Face Pipelines, it wraps Engine with pre-processing and post-processing steps, allowing you to make requests on raw data and receive post-processed predictions.

- **Server** is a REST API wrapper around Pipelines built on FastAPI and Uvicorn. It enables you to start a model serving endpoint running DeepSparse with a single CLI.

This directory offers examples using each API in various supported tasks. 

### Supported Tasks

DeepSparse supports the following tasks out of the box:

|          NLP          |            CV             |
|-----------------------|---------------------------|
| [Text Classification `"text-classification"`](use-cases/nlp/text-classification.md)     | [Image Classification `"image_classification"`](use-cases/cv/image-classification.md)     |
| [Token Classification `"token-classification"`](use-cases/nlp/token-classification.md)  | [Object Detection `"yolo"`](use-cases/cv/object-detection-yolov5.md)    |
| [Sentiment Analysis `"sentiment-analysis"`](use-cases/nlp/sentiment-analysis.md)        | [Instance Segmentation `"yolact"`](image-segmentation-yolact.md)        |
| [Question Answering `"question-answering"`](use-cases/nlp/question-answering.md)        |                                                                         |
| [Zero-Shot Text Classification `"zero-shot-text-classification"`](use-cases/nlp/zero-shot-text-classification.md) |                                               |
| [Embedding Extraction `"transformers_embedding_extraction"`](use-cases/nlp/transformers-embedding-extraction.md) |                                               |

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
url = 'http://0.0.0.0:5543/predict'

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
