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


<div style="display: flex; flex-direction: column; align-items: center;">
  <h1>
    <img alt="tool icon" src="https://raw.githubusercontent.com/neuralmagic/deepsparse/main/docs/old/source/icon-deepsparse.png" />
    &nbsp;&nbsp;DeepSparse
  </h1>
  <h4>Sparsity-aware deep learning inference runtime for CPUs</h4>
  <div align="center">
  <a href="https://docs.neuralmagic.com/deepsparse/">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-darkred?&style=for-the-badge&logo=read-the-docs" height="20" />
  </a>
  <a href="https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ/">
    <img alt="Slack" src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height="20" />
  </a>
  <a href="https://github.com/neuralmagic/deepsparse/issues/">
    <img alt="Support" src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height="20" />
  </a>
  <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
    <img alt="YouTube" src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height="20" />
  </a>
  <a href="https://twitter.com/neuralmagic">
    <img alt="Twitter" src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height="20" />
  </a>
</div>

[DeepSparse](https://github.com/neuralmagic/deepsparse) is a CPU inference runtime that takes advantage of sparsity to accelerate neural network inference. Coupled with [SparseML](https://github.com/neuralmagic/sparseml), our optimization library for pruning and quantizing your models, DeepSparse delivers exceptional inference performance on CPU hardware.

<p align="center">
   <img alt="NM Flow" src="https://github.com/neuralmagic/deepsparse/assets/3195154/51e62fe7-9d9a-4fa5-a774-877158da1e29" width="60%" />
</p>

## ✨NEW✨ DeepSparse LLMs

Neural Magic is excited to announce initial support for performant LLM inference in DeepSparse with:
- sparse kernels for speedups and memory savings from unstructured sparse weights.
- 8-bit weight and activation quantization support.
- efficient usage of cached attention keys and values for minimal memory movement.

![mpt-chat-comparison](https://github.com/neuralmagic/deepsparse/assets/3195154/ccf39323-4603-4489-8462-7b103872aeb3)

### Try It Now

Install (requires Linux):
```bash
pip install -U deepsparse-nightly[llm]
```

Run inference:
```python
from deepsparse import TextGeneration
pipeline = TextGeneration(model="zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized")

prompt="""
Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is sparsity? ### Response:
"""
print(pipeline(prompt, max_new_tokens=75).generations[0].text)

# Sparsity is the property of a matrix or other data structure in which a large number of elements are zero and a smaller number of elements are non-zero. In the context of machine learning, sparsity can be used to improve the efficiency of training and prediction.
```

> [Check out the `TextGeneration` documentation for usage details.](https://github.com/neuralmagic/deepsparse/blob/main/docs/llms/text-generation-pipeline.md)

### Sparsity :handshake: Performance

Developed in collaboration with IST Austria, [our recent paper](https://arxiv.org/abs/2310.06927) details a new technique called **Sparse Fine-Tuning**, which allows us to prune MPT-7B to 60% sparsity during fine-tuning without drop in accuracy. With our new support for LLMs, DeepSparse accelerates the sparse-quantized model 7x over the dense baseline:

<div align="center">
    <img src="https://github.com/neuralmagic/deepsparse/assets/3195154/8687401c-f479-4999-ba6b-e01c747dace9" width="60%"/>
</div>

> [Learn more about our Sparse Fine-Tuning research.](https://github.com/neuralmagic/deepsparse/blob/main/research/mpt#sparse-finetuned-llms-with-deepsparse)

> [Check out the model running live on Hugging Face.](https://huggingface.co/spaces/neuralmagic/sparse-mpt-7b-gsm8k)

### LLM Roadmap

Following this initial launch, we are rapidly expanding our support for LLMs, including:

1. Productizing Sparse Fine-Tuning: Enable external users to apply sparse fine-tuning to their datasets via SparseML.
2. Expanding model support: Apply our sparse fine-tuning results to Llama 2 and Mistral models.
3. Pushing for higher sparsity: Improving our pruning algorithms to reach even higher sparsity.

## Computer Vision and NLP Models

In addition to LLMs, DeepSparse supports many variants of CNNs and Transformer models, such as BERT, ViT, ResNet, EfficientNet, YOLOv5/8, and many more! Take a look at the [Computer Vision](https://sparsezoo.neuralmagic.com/?modelSet=computer_vision) and [Natural Language Processing](https://sparsezoo.neuralmagic.com/?modelSet=natural_language_processing) domains of [SparseZoo](https://sparsezoo.neuralmagic.com/), our home for optimized models.

### Installation

Install via [PyPI](https://pypi.org/project/deepsparse/) ([optional dependencies detailed here](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/installation.md)):

```bash
pip install deepsparse 
```

To experiment with the latest features, there is a nightly build available using `pip install deepsparse-nightly` or you can clone and install from source using `pip install -e path/to/deepsparse`.

#### System Requirements
- Hardware: [x86 AVX2, AVX-512, AVX-512 VNNI and ARM v8.2+](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/hardware-support.md)
- Operating System: Linux
- Python: 3.8-3.11
- ONNX versions 1.5.0-1.15.0, ONNX opset version 11 or higher

For those using Mac or Windows, we recommend using Linux containers with Docker.

## Deployment APIs

DeepSparse includes three deployment APIs:

- **Engine** is the lowest-level API. With Engine, you compile an ONNX model, pass tensors as input, and receive the raw outputs.
- **Pipeline** wraps the Engine with pre- and post-processing. With Pipeline, you pass raw data and receive the prediction.
- **Server** wraps Pipelines with a REST API using FastAPI. With Server, you send raw data over HTTP and receive the prediction.

### Engine

The example below downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo, compiles the model, and runs inference on randomly generated input. Users can provide their own ONNX models, whether dense or sparse.

```python
from deepsparse import Engine

# download onnx, compile
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
compiled_model = Engine(model=zoo_stub, batch_size=1)

# run inference (input is raw numpy tensors, output is raw scores)
inputs = compiled_model.generate_random_inputs()
output = compiled_model(inputs)
print(output)

# > [array([[-0.3380675 ,  0.09602544]], dtype=float32)] << raw scores
```

### Pipeline

Pipelines wrap Engine with pre- and post-processing, enabling you to pass raw data and receive the post-processed prediction. The example below downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo, sets up a pipeline, and runs inference on sample data.

```python
from deepsparse import Pipeline

# download onnx, set up pipeline
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"  
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=zoo_stub,          # zoo stub or path to local onnx file
)

# run inference (input is a sentence, output is the prediction)
prediction = sentiment_analysis_pipeline("I love using DeepSparse Pipelines")
print(prediction)
# > labels=['positive'] scores=[0.9954759478569031]
```

### Server

Server wraps Pipelines with REST APIs, enabling you to set up a model-serving endpoint running DeepSparse. This enables you to send raw data to DeepSparse over HTTP and receive the post-processed predictions. DeepSparse Server is launched from the command line and configured via arguments or a server configuration file. The following downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo and launches a sentiment analysis endpoint:

```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Sending a request:

```python
import requests

url = "http://localhost:5543/v2/models/sentiment_analysis/infer" # Server's port default to 5543
obj = {"sequences": "Snorlax loves my Tesla!"}

response = requests.post(url, json=obj)
print(response.text)
# {"labels":["positive"],"scores":[0.9965094327926636]}
```

### Additional Resources
- [Use Cases Page](https://github.com/neuralmagic/deepsparse/tree/main/docs/use-cases) for more details on supported tasks
- [Pipelines User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/deepsparse-pipelines.md) for Pipeline documentation
- [Server User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/deepsparse-server.md) for Server documentation
- [Benchmarking User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/deepsparse-benchmarking.md) for benchmarking documentation
- [Cloud Deployments and Demos](https://github.com/neuralmagic/deepsparse/tree/main/examples/)
- [User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide) for more detailed documentation


## Product Usage Analytics

DeepSparse gathers basic usage telemetry, including, but not limited to, Invocations, Package, Version, and IP Address, for Product Usage Analytics purposes. Review Neural Magic's [Products Privacy Policy](https://neuralmagic.com/legal/) for further details on how we process this data. 

To disable Product Usage Analytics, run:
```bash
export NM_DISABLE_ANALYTICS=True
```

Confirm that telemetry is shut off through info logs streamed with engine invocation by looking for the phrase "Skipping Neural Magic's latest package version check."

## Community

### Get In Touch

- [Contribution Guide](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md)
- [Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues) 
- [Subscribe To Our Newsletter](https://neuralmagic.com/subscribe/)
- [Blog](https://www.neuralmagic.com/blog/) 

For more general questions about Neural Magic, [complete this form.](http://neuralmagic.com/contact/)

### License

- **DeepSparse Community** is free to use and is licensed under the [Neural Magic DeepSparse Community License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC)
Some source code, example files, and scripts included in the DeepSparse GitHub repository or directory are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.

- **DeepSparse Enterprise** requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

### Cite

Find this project useful in your research or other communications? Please consider citing:

```bibtex
@misc{kurtic2023sparse,
      title={Sparse Fine-Tuning for Inference Acceleration of Large Language Models}, 
      author={Eldar Kurtic and Denis Kuznedelev and Elias Frantar and Michael Goin and Dan Alistarh},
      year={2023},
      url={https://arxiv.org/abs/2310.06927},
      eprint={2310.06927},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{kurtic2022optimal,
      title={The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models}, 
      author={Eldar Kurtic and Daniel Campos and Tuan Nguyen and Elias Frantar and Mark Kurtz and Benjamin Fineran and Michael Goin and Dan Alistarh},
      year={2022},
      url={https://arxiv.org/abs/2203.07259},
      eprint={2203.07259},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@InProceedings{
    pmlr-v119-kurtz20a, 
    title = {Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks}, 
    author = {Kurtz, Mark and Kopinsky, Justin and Gelashvili, Rati and Matveev, Alexander and Carr, John and Goin, Michael and Leiserson, William and Moore, Sage and Nell, Bill and Shavit, Nir and Alistarh, Dan}, 
    booktitle = {Proceedings of the 37th International Conference on Machine Learning}, 
    pages = {5533--5543}, 
    year = {2020}, 
    editor = {Hal Daumé III and Aarti Singh}, 
    volume = {119}, 
    series = {Proceedings of Machine Learning Research}, 
    address = {Virtual}, 
    month = {13--18 Jul}, 
    publisher = {PMLR}, 
    pdf = {http://proceedings.mlr.press/v119/kurtz20a/kurtz20a.pdf},
    url = {http://proceedings.mlr.press/v119/kurtz20a.html}
}

@article{DBLP:journals/corr/abs-2111-13445,
  author    = {Eugenia Iofinova and Alexandra Peste and Mark Kurtz and Dan Alistarh},
  title     = {How Well Do Sparse Imagenet Models Transfer?},
  journal   = {CoRR},
  volume    = {abs/2111.13445},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.13445},
  eprinttype = {arXiv},
  eprint    = {2111.13445},
  timestamp = {Wed, 01 Dec 2021 15:16:43 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-13445.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# All Thanks To Our Contributors

<a href="https://github.com/neuralmagic/deepsparse/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=neuralmagic/deepsparse" />
</a>
