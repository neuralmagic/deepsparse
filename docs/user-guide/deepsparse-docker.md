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
# How to Use DeepSparse With Docker 
DeepSparse is an efficient and powerful tool for running inference on sparse and quantized models. Apart from installing DeepSparse with `pip`, it can be easily set up using [Docker](https://www.docker.com/) which allows you to start using DeepSparse without having to manually install all the required dependencies.

In this guide, you will learn how to use DeepSparse with Docker for various use cases, such as running an HTTP server, working with the `Engine`, using the `Pipeline`, and benchmarking DeepSparse's performance.

## Prerequisites

Before you begin, make sure you have Docker installed on your machine. You can download and install it from the [official Docker website](https://www.docker.com/products/docker-desktop).

## Pulling and Tagging the DeepSparse Docker Image

First, pull the `deepsparse` image from the GitHub Container Registry:```

```bash
docker pull ghcr.io/neuralmagic/deepsparse:1.4.2
```

Tag the image to make it easier to reference later:

```bash
docker tag ghcr.io/neuralmagic/deepsparse:1.4.2 deepsparse_docker
```
## DeepSparse Server Example

DeepSparse Server, built on the popular FastAPI and Uvicorn stack, allows you to set up a REST endpoint for serving inferences over HTTP. It wraps the Pipeline API, inheriting all the utilities provided by Pipelines.

Start the `deepsparse` container in interactive mode and publish the containers port 5543 to the local machine's port 5543 to expose the port outside the container. 

Here's the meaning of the commands after  `docker container run`:
- `i` Keeps STDIN open even if not attached
- `t` to allocate a pseudo-TTY
- `p` publishes Docker's internal port 5543 to the local machines port 5543
```bash
docker container run -it -p 5543:5543 deepsparse_docker
```
Running the following CLI command inside the container launches a sentiment analysis pipeline with a 90% pruned-quantized BERT model identified by its SparseZoo stub:

```bash
deepsparse.server --task sentiment_analysis --model_path "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned90-none"
```
Alternatively, you can run the two commands in a single line: 
```bash 
docker container run -p 5543:5543 deepsparse_docker deepsparse.server --task sentiment_analysis --model_path "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned90-none"
```
<!-- markdown-link-check-disable -->
You should see Uvicorn report that it is running on http://0.0.0.0:5543. Once launched, a `/docs` path is created with full endpoint descriptions and support for making sample requests.
<!-- markdown-link-check-enable -->

Here is an example client request, using the Python requests library for formatting the HTTP:
```python
import requests

url = "http://localhost:5543/predict"

obj = {
    "sequences": "Who is Mark?",
}

response = requests.post(url, json=obj)
response.content
# b'{"labels":["negative"],"scores":[0.9695534706115723]}'
```
## DeepSparse Engine example

Engine is the lowest-level API for interacting with DeepSparse. As much as possible, we recommended you use the Pipeline API but Engine is available if you want to handle pre- or post-processing yourself.

With Engine, we can compile an ONNX file and run inference on raw tensors.

Here's an example, using a 90% pruned-quantized BERT trained on SST2 from SparseZoo.

Save this script in a file named `app.py`:
```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path

def run():
    # download onnx from sparsezoo and compile with batchsize 1
    sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
    batch_size = 1
    bert_engine = Engine(
    model=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
    batch_size=batch_size   # defaults to batch size 1
    )

    # input is raw numpy tensors, output is raw scores for classes
    inputs = generate_random_inputs(model_to_path(sparsezoo_stub), batch_size)
    output = bert_engine(inputs)
    print(output)


if __name__ == "__main__":
    run()
```
Next create a Dockerfile. The name of the file should be `Dokcerfile`. This file has instructions for: 
- Pulling the DeepSparse Docker Image 
- Copying the Python script into the container 
- Running the Python script 
```Dockerfile
FROM ghcr.io/neuralmagic/deepsparse:1.4.2

# Set the working directory to the user's home directory
WORKDIR /app

# Copy the current directory contents into the container 
COPY . .

#Run the Python script
CMD ["python", "app.py"]
```
Create a DeepSparse Container where the Python script will run in. The `-t` argument tags the container with the given name. 

Run the following command in the directory containing the `Dockerfle` and `app.py`. 
```bash 
docker build -t engine_deepsparse_docker .
```
Run your newly created DeepSparse Container: 
```bash 
docker container run  engine_deepsparse_docker
# [array([[-0.34614536,  0.09025408]], dtype=float32)]
```

## DeepSparse Pipeline Example
Pipeline is the default interface for interacting with DeepSparse.

Similar to Hugging Face Pipelines, DeepSparse Pipelines wrap pre- and post-processing around the inference performed by the Engine. 
This creates a clean API that allows you to pass raw images and text to DeepSparse and receive the post-processed prediction, making it easy to add DeepSparse to your application.

Use the `Pipeline.create()` constructor to create an instance of a sentiment analysis Pipeline with a 90% pruned-quantized version of BERT trained on SST2. We can then pass the Pipeline raw text and receive the predictions. 
All the pre-processing (such as tokenizing the input) is handled by the Pipeline.

Save this script in a file called `app.py`:
```python
from deepsparse import Pipeline

def run():
    # download onnx from sparsezoo and compile with batch size 1
    sparsezoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
    batch_size = 1
    sa_pipeline = Pipeline.create(
    task="sentiment-analysis",
    model_path=sparsezoo_stub,   # sparsezoo stub or path to local ONNX
    batch_size=1                 # default batch size is 1
    )

    # run inference on image file
    prediction = sa_pipeline("The sentiment analysis pipeline is fast and easy to use")
    print(prediction)


if __name__ == "__main__":
    run()
```
Next create a Dockerfile. The file should be named `Dockerfile`: 
```Dockerfile
FROM ghcr.io/neuralmagic/deepsparse:1.4.2

# Set the working directory to the user's home directory
WORKDIR /app

# Copy the current directory contents into the container 
COPY . .

#Run the Python script
CMD ["python", "app.py"]
```

Create Docker Container using the Dockerfile. The `Dockerfile` and `app.py` should be in the same folder. Run the following command in that folder:
```bash 
docker build -t pipeline_deepsparse_docker .
```
Run the Docker Container: 
```bash 
docker container run  pipeline_deepsparse_docker
# labels=['positive'] scores=[0.9955807328224182]
```
## DeepSparse Benchmarking

Use the benchmarking utility to check the DeepSparse's performance: 
```bash
docker container run -it deepsparse_docker deepsparse.benchmark "zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned90-none"

> Original Model Path: zoo:nlp/sentiment_analysis/distilbert-none/pytorch/huggingface/sst2/pruned90-none
> Batch Size: 1
> Scenario: sync
> Throughput (items/sec): 1.4351
> Latency Mean (ms/batch): 696.7735
> Latency Median (ms/batch): 687.1720
> Latency Std (ms/batch): 465.9775
> Iterations: 15
```

