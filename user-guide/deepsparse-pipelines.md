# DeepSparse Pipelines

Pipelines are the default API for deploying a model with DeepSparse. 

Similar to Hugging Face Pipelines, DeepSparse Pipelines wrap inference with task-specific
pre- and post-processing, enabling you to pass raw data and recieve the predictions.

## Quickstart

Let's try a quick example of the Pipeline API. All we have to do is pass a task and model to the
the `Pipeline.create` function, and then we can run inference on raw data using DeepSparse!

This example creates a sentiment analysis Pipeline with a 90% pruned-quantized verion of BERT
from the SparseZoo.

```python
from deepsparse import Pipeline

# download and compile onnx, create pipeline 
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=zoo_stub,          # zoo stub or path to local onnx file
)

# run inference
print(sentiment_analysis_pipeline("I love using DeepSparse Pipelines"))
# >>> labels=['positive'] scores=[0.9954759478569031]
```

In this case we passed a SparseZoo stub as the model, which instructs DeepSparse to download the 
relevant ONNX file from the SparseZoo. To deploy your own model, pass a path to a `model.onnx` file or to a 
folder containing `model.onnx` and supporting files (e.g., the Hugging Face `tokenizer.json` and `config.json`).

## Supported Use Cases

Pipelines support many CV and NLP use cases out of the box. Check out the use case pages for more details on task-specific APIs.

**Computer Vision**:
- Image Classification: `task="image_classification"`
- Object Detection: `task="yolo"`
- Instance Segmentation: `task="yolact"`

**Natural Language Processing**:
- Embedding Extraction:`task="transformers_embedding_extraction"`
- Text Classification: `task="text-classification"`
- Zero Shot Text Classification: `task="zero-shot-text-classification"` 
- Sentiment Analysis: `task="sentiment-analysis"`
- Token Classification: `task="token-classification"`
- Question Answering: `task="question-answering"`

## Custom Use Case

Beyond officially supported use cases, Pipelines can be extended to additional tasks via the 
`CustomTaskPipeline`.

`CustomTaskPipelines` are passed the following arguments:
- `model_path` - a SparseZoo stub or path to a local ONNX file
- `process_inputs_fn` - an optional function that handles pre-processing of input into a list 
of numpy arrays that can be passed directly to the inference forward pass
- `process_outputs_fn` - an optional function that handles post-processing of the list of numpy arrays 
that are the output of the engine forward pass 

Let's demonstrate an example of how we could replicate the functionality of the image classification
pipeline as a custom Pipeline.

Download an image and ONNX file (a 95% pruned-quantized ResNet-50) for the demo:
```
wget https://raw.githubusercontent.com/neuralmagic/docs/main/files-for-examples/use-cases/embedding-extraction/goldfish.jpg
sparsezoo.download zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none --save-dir ./resnet-50-pruned-quant
```

We can create a custom image classification Pipeline which returns the raw logits and class probabilities 
for the 1000 ImageNet classes with the following:

For the purposes of this quick example, make sure you have `torch` `torchvision` and `Pillow` installed.
```python
from deepsparse.pipelines.custom_pipeline import CustomTaskPipeline
from torchvision import transforms
from PIL import Image
import numpy as np
import torch

IMAGENET_RGB_MEANS = [0.485, 0.456, 0.406]
IMAGENET_RGB_STDS = [0.229, 0.224, 0.225]
preprocess_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
])

def preprocess(img_file):
    with open(img_file, "rb") as img_file:
        img = Image.open(img_file)
        img = img.convert("RGB")
    img = preprocess_transforms(img)
    batch = torch.stack([img])
    return [batch.numpy()] 

custom_pipeline = CustomTaskPipeline(
    model_path="./resnet-50-pruned-quant/model.onnx",
    process_inputs_fn=preprocess,
)

scores, probs = custom_pipeline("goldfish.jpg")

print(scores.shape)
print(probs.shape)
print(np.sum(probs))
print(np.argmax(probs))

# >> (1,1000)
# >> (1,1000)
# >> ~1.00000
# >> 1 << index of the goldfish class in ImageNet
```

## Pipeline Utilities

Beyond supporting pre- and post-processing, Pipelines also offer additional utilities that simplify
the deployment process.

### Batch Size

The `batch_size` argument configures the batch size of the Pipeline, modifying the underlying ONNX graph for you. 
The default is batch size 1, and but we can override to batch size 3 with the following:

```python
from deepsparse import Pipeline

zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
batch_size = 3
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=zoo_stub,          # zoo stub or path to local onnx file
  batch_size=batch_size         # default is batch 1
)

sentences = [
    "I love DeepSparse Pipelines",
    "I hated changing the batch size with my prior Deep Learning framework",
    "DeepSparse makes it very easy to adjust the batch size"
]
output = sentiment_analysis_pipeline(sentences)
print(output)

# >>> labels=['positive', 'negative', 'positive'] scores=[0.9969560503959656, 0.9964107871055603, 0.7127435207366943]
```

### Number of Cores

The `num_cores` argument configures the number of physical cores used by DeepSparse. The default is None, which
instructs DeepSparse to use all physical cores available on the system. We can override to use only
one core with the following:

```python
from deepsparse import Pipeline

zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=zoo_stub,          # zoo stub or path to local onnx file
  num_cores=1
)

sentences = "I love how DeepSparse makes it easy to configure the number of cores used"

output = sentiment_analysis_pipeline(sentences)
print(output)

# >> labels=['positive'] scores=[0.9951152801513672] << but runs slower than if using all cores
```

### Dynamic Batch Size

We can utilize an the multi-stream capabilites of DeepSparse to make requests with dynamic batch sizes.

Let's create an example with a single sentiment analysis Pipeline with dynamic batch sizes by 
setting the `batch_size` argument to None. Under the hood, the pipeline will split the batch into 
multiple asynchronous requests using the multi-stream scheduler.

```python
from deepsparse import Pipeline

zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
sentiment_analysis_pipeline = Pipeline.create(
    task="sentiment_analysis",
    model_path=zoo_stub,
    batch_size=None,        # setting to None enables dynamic batch
)

b1_request = ["This multi model concept is great!"]
b4_request = b1_request * 4

output_b1 = sentiment_analysis_pipeline(b1_request)
output_b4 = sentiment_analysis_pipeline(b4_request)

print(output_b1)
# >> labels=['positive'] scores=[0.9995297789573669]

print(output_b4)
# >> labels=['positive', 'positive', 'positive', 'positive'] scores=[0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669]
```

### Deploy Multiple Models on the Same System

Some deployment scenarios will require running multiple instances of DeepSparse on a single 
machine. DeepSparse includes a concepts called Context. Contexts can be used to run multiple 
models with the same scheduler, enabling DeepSparse to manage the resources of the system effectively, 
keeping engines that are running different models from fighting over resources.

Let's create an example with multiple sentiment analysis Pipelines, one with batch size 1 (for maximum latency) 
and one with batch size 32 (for maximum throughput).

```python
from concurrent.futures import ThreadPoolExecutor
from deepsparse.engine import Context
from deepsparse import Pipeline

context = Context()
executor = ThreadPoolExecutor(max_workers=context.num_streams)

zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  

sentiment_analysis_pipeline_b1 = Pipeline.create(
    task="sentiment_analysis",
    model_path=zoo_stub,
    batch_size=1,
    context=context,
    executor=executor
)

sentiment_analysis_pipeline_b32 = Pipeline.create(
    task="sentiment_analysis",
    model_path=zoo_stub,
    batch_size=32,
    context=context,
    executor=executor
)

b1_request = ["This multi model concept is great!"]
b32_request = b1_request * 32

output_b1 = sentiment_analysis_pipeline_b1(b1_request)
output_b32 = sentiment_analysis_pipeline_b32(b32_request)

print(output_b1)
print(output_b32)

# >> labels=['positive'] scores=[0.9995297789573669]
# >> labels=['positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive', 'positive'] scores=[0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669, 0.9995297789573669]
```

If you are deploying multiple models on a same system, you may want to answer multiple
requests concurrently. We can enable this but setting the `num_streams` argument in the Context argument.

```python
from concurrent.futures import ThreadPoolExecutor
from deepsparse.engine import Context
from deepsparse.pipeline import Pipeline
import threading

class ExecutorThread(threading.Thread):
    def __init__(self, pipeline, input, iters=1):
        super(ExecutorThread, self).__init__()
        self.pipeline = pipeline
        self.input = input
        self.iters = iters
        
    def run(self):
        for _ in range(self.iters):
            output = self.pipeline(self.input)
            print(output)

num_concurrent_requests = 2

context = Context(num_streams=num_concurrent_requests)
executor = ThreadPoolExecutor(max_workers=context.num_streams)

zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  

sentiment_analysis_pipeline_b1 = Pipeline.create(
    task="sentiment_analysis",
    model_path=zoo_stub,
    batch_size=1,
    context=context,
    executor=executor
)

sentiment_analysis_pipeline_b32 = Pipeline.create(
    task="sentiment_analysis",
    model_path=zoo_stub,
    batch_size=32,
    context=context,
    executor=executor
)

b1_request = ["This multi model concept is great!"]
b64_request = b1_request * 32

threads = [
    ExecutorThread(sentiment_analysis_pipeline_b1, input=b1_request, iters=64),
    ExecutorThread(sentiment_analysis_pipeline_b32, input=b64_request, iters=1),
]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

# mutiple b=1 queries print results before the b=32 query returns
```

Note that requests will be execute in a FIFO manner, with a maximum of `num_concurrent_requests` running at once.
As a result, high traffic on one of your Pipelines can impact performance on the other Pipeline. If you prefer to 
isolate your Pipelines, we recommend using an orchestration framework such as Docker and Kubernetes with 
one DeepSparse Pipeline running in each container for proper process isolation.

### Multi-Stream Scheduling

Stay tuned for documentation on enabling multi-stream scheduling with DeepSparse Pipelines.

### Logging

Stay tuned for documentation on enabling logging with DeepSparse Pipelines.
