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
    <a href="https://github.com/neuralmagic/deepsparse/actions/workflows/quality-check.yaml">
      <img alt="Main" src="https://img.shields.io/github/workflow/status/neuralmagic/deepsparse/Quality%20Checks/main?label=build&style=for-the-badge" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/releases">
      <img alt="GitHub release" src="https://img.shields.io/github/release/neuralmagic/deepsparse.svg?style=for-the-badge" height="20" />
    </a>
    <a href="https://github.com/neuralmagic/deepsparse/blob/main/CODE_OF_CONDUCT.md">
      <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.1%20adopted-ff69b4.svg?color=yellow&style=for-the-badge" height="20" />
    </a>
    <a href="https://www.youtube.com/channel/UCo8dO_WMGYbWCRnj_Dxr4EA">
      <img alt="YouTube" src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height="20" />
    </a>
    <a href="https://medium.com/limitlessai">
      <img alt="Medium" src="https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white" height="20" />
    </a>
    <a href="https://twitter.com/neuralmagic">
      <img alt="Twitter" src="https://img.shields.io/twitter/follow/neuralmagic?color=darkgreen&label=Follow&style=social" height="20" />
    </a>
  </div>
</div>

[DeepSparse](https://github.com/neuralmagic/deepsparse) is a CPU inference runtime that takes advantage of sparsity to accelerate neural network inference. Coupled with SparseML, our optimization library for pruning and quantizing your models, DeepSparse delivers exceptional performance on commodity hardware. [Check out SparseML for details of training sparse models](https://github.com/neuralmagic/sparseml).

<p align="center">
   <img alt="NM Flow" src="https://github.com/neuralmagic/deepsparse/blob/7ee5e60f13b1fd321c5282c91e2873b3363ec911/docs/neural-magic-workflow.png" width="60%" />
</p>


## ✨NEW✨ DeepSparse LLMs

We are pleased to announce initial support for LLMs in DeepSparse, starting with [MosaicML's MPT-7b](https://github.com/neuralmagic/deepsparse/tree/main/research/mpt).

Install:
```bash
pip install -U deepsparse-nightly[transformers]==1.6.0.20231007
```

Run inference:
```python
from deepsparse import TextGeneration
pipeline = TextGeneration(model="zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized")

prompt="""
Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is sparsity? ### Response:
"""

print(pipeline(prompt, max_new_tokens=75).generations[0].text)
### >> Sparsity is the property of a matrix or other data structure in which a large number of elements are zero, and a smaller number of elements are non-zero. In the context of machine learning, sparsity can be used to improve the efficiency of training and prediction.
```

DeepSparse is optimized for state-of-the-art text generation latency for LLMs with:
- Optimized sparse quantized x86 and ARM CPU kernels
- Efficient usage of cached attention keys and values for minimal memory movement
- Compressed memory using sparse weights
- Run locally or in the cloud on Linux (Mac coming soon!)

- [Check out our LLM documentation for more details]().

### Performance / Accuracy

Using our state-of-the-art **Sparse Finetuning** technique, we pruned MPT-7B to 60% sparsity (with INT8 quantization). With DeepSparse, the 60% sparse-INT8 model runs ~7x faster than the dense-FP32 baseline with no drop in accuracy:

<div align="center">
    <img src="https://github.com/neuralmagic/deepsparse/assets/3195154/8687401c-f479-4999-ba6b-e01c747dace9" width="50%"/>
</div>

- [Check out the current status of our **Sparse Finetuning** research](https://github.com/neuralmagic/deepsparse/tree/llm-docs-2/research/mpt#sparse-finetuned-llms-with-deepsparse)

### LLM Roadmap

Following this initial launch, we are rapidly expanding our support for LLMs, including:

1. **Productizing Sparse Finetuning**: Enable external users to apply the sparse fine-tuning to their datasets via SparseML
2. **Expanding Model Support**: Apply our sparse fine-tuning results to Llama2 and Mistral models
3. **Pushing to Higher Sparsity**: Improving our pruning algorithms to reach higher sparsity

## Pre-LLM Support

### Installation

DeepSparse can be install via PyPI. We recommend using a virtual enviornment.

```bash
pip install deepsparse
```
[Check out the Installation page for optional dependencies](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/installation.md).

- Hardware: x86 AVX2, AVX512, AVX512-VNNI and ARM v8.2+.
- Operating System: Linux (manylinux compliant systems)
- Python: v3.8-3.10
- ONNX versions 1.5.0-1.12.0, ONNX opset version 11 or higher

For those using Mac or Windows, running Linux in a Docker or virtual machine is necessary to use DeepSparse.

### Deployment APIs

DeepSparse includes three deployment APIs:

- **Engine** is the lowest-level API. With Engine, you pass tensors and receive the raw logits.
- **Pipeline** wraps the Engine with pre- and post-processing. With Pipeline, you pass raw data and receive the prediction.
- **Server** wraps Pipelines with a REST API using FastAPI. With Server, you send raw data over HTTP and receive the prediction.

#### Engine

The example below downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo, compiles the model, and runs inference on randomly generated input.

```python
from deepsparse import Engine
from deepsparse.utils import generate_random_inputs, model_to_path

# download onnx, compile
zoo_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
batch_size = 1
compiled_model = Engine(model=zoo_stub, batch_size=batch_size)

# run inference (input is raw numpy tensors, output is raw scores)
inputs = generate_random_inputs(model_to_path(zoo_stub), batch_size)
output = compiled_model(inputs)
print(output)

# > [array([[-0.3380675 ,  0.09602544]], dtype=float32)] << raw scores
```

#### Pipeline

Pipeline is the default API for interacting with DeepSparse. Similar to Hugging Face Pipelines, DeepSparse Pipelines wrap Engine with pre- and post-processing (as well as other utilities), enabling you to send raw data to DeepSparse and receive the post-processed prediction. The example below downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo, sets up a pipeline, and runs inference on sample data.

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

#### Additional Resources 
- Check out the [Use Cases Page](https://github.com/neuralmagic/deepsparse/tree/main/docs/use-cases) for more details on supported tasks.
- Check out the [Pipelines User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/deepsparse-pipelines.md) for more usage details.

### Server

Server wraps Pipelines with REST APIs, enabling you to stand up model serving endpoint running DeepSparse. This enables you to send raw data to DeepSparse over HTTP and receive the post-processed predictions. DeepSparse Server is launched from the command line, configured via arguments or a server configuration file. The following downloads a 90% pruned-quantized BERT model for sentiment analysis in ONNX format from SparseZoo and launches a sentiment analysis endpoint:

```bash
deepsparse.server \
  --task sentiment-analysis \
  --model_path zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Sending a request:

```python
import requests

url = "http://localhost:5543/predict" # Server's port default to 5543
obj = {"sequences": "Snorlax loves my Tesla!"}

response = requests.post(url, json=obj)
print(response.text)
# {"labels":["positive"],"scores":[0.9965094327926636]}
```

### Additional Resources
- [User Guide](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide)
- [Use Cases](https://github.com/neuralmagic/deepsparse/tree/main/docs/use-cases)
- [Benchmarking Performance](https://github.com/neuralmagic/deepsparse/tree/main/docs/user-guide/deepsparse-benchmarking.md)
- [Cloud Deployments and Demos](https://github.com/neuralmagic/deepsparse/tree/main/examples/)

### Versions
- [DeepSparse](https://pypi.org/project/deepsparse) | stable
- [DeepSparse-Nightly](https://pypi.org/project/deepsparse-nightly/) | nightly (dev)
- [GitHub](https://github.com/neuralmagic/deepsparse/releases) | releases

## Product Usage Analytics
DeepSparse Community Edition gathers basic usage telemetry including, but not limited to, Invocations, Package, Version, and IP Address for Product Usage Analytics purposes. Review Neural Magic's [Products Privacy Policy](https://neuralmagic.com/legal/) for further details on how we process this data.

To disable Product Usage Analytics, run the command:
```bash
export NM_DISABLE_ANALYTICS=True
```

Confirm that telemetry is shut off through info logs streamed with engine invocation by looking for the phrase "Skipping Neural Magic's latest package version check." For additional assistance, reach out through the [DeepSparse GitHub Issue queue](https://github.com/neuralmagic/deepsparse/issues).

## Community

### Be Part of the Future... And the Future is Sparse!

- [Contribution Guide](https://github.com/neuralmagic/deepsparse/blob/main/CONTRIBUTING.md)
- [Community Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
- [GitHub Issue Queue](https://github.com/neuralmagic/deepsparse/issues) 
- [Subscribe To Our Newsletter](https://neuralmagic.com/subscribe/) to the Neural Magic community.

For more general questions about Neural Magic, complete this [form.](http://neuralmagic.com/contact/)

### License

[DeepSparse Community](https://docs.neuralmagic.com/products/deepsparse) is licensed under the [Neural Magic DeepSparse Community License.](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE-NEURALMAGIC)
Some source code, example files, and scripts included in the deepsparse GitHub repository or directory are licensed under the [Apache License Version 2.0](https://github.com/neuralmagic/deepsparse/blob/main/LICENSE) as noted.

[DeepSparse Enterprise](https://docs.neuralmagic.com/products/deepsparse-ent) requires a Trial License or [can be fully licensed](https://neuralmagic.com/legal/master-software-license-and-service-agreement/) for production, commercial applications.

### Cite

Find this project useful in your research or other communications? Please consider citing:

```bibtex
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
  author    = {Eugenia Iofinova and
               Alexandra Peste and
               Mark Kurtz and
               Dan Alistarh},
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
