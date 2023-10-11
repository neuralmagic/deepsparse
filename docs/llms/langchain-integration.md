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

# **DeepSparse LangChain Integration**

[DeepSparse](https://github.com/neuralmagic/deepsparse) has an official integration within [LangChain](https://python.langchain.com/docs/integrations/llms/deepsparse).
It is broken into two parts: installation and then examples of DeepSparse usage.

## Installation and Setup

- Install the Python packages with `pip install deepsparse-nightly langchain`
- Choose a [SparseZoo model](https://sparsezoo.neuralmagic.com/?useCase=text_generation) or export a support model to ONNX [using Optimum](https://github.com/neuralmagic/notebooks/blob/main/notebooks/opt-text-generation-deepsparse-quickstart/OPT_Text_Generation_DeepSparse_Quickstart.ipynb)
- Models hosted on HuggingFace are also supported by prepending `"hf:"` to the model id, such as [`"hf:mgoin/TinyStories-33M-quant-deepsparse"`](https://huggingface.co/mgoin/TinyStories-33M-quant-deepsparse)

## Wrappers

There exists a DeepSparse LLM wrapper, which you can access with:

```python
from langchain.llms import DeepSparse
```

It provides a simple, unified interface for all models:

```python
from langchain.llms import DeepSparse
llm = DeepSparse(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none')
print(llm('def fib():'))
```

And provides support for per token output streaming:

```python
from langchain.llms import DeepSparse
llm = DeepSparse(
    model="zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base_quant-none",
    streaming=True
)
for chunk in llm.stream("Tell me a joke", stop=["'","\n"]):
    print(chunk, end='', flush=True)
```

## Configuration

It has arguments to control the model loaded, any configs for how the model should be loaded, configs to control how tokens are generated, and then whether to return all tokens at once or to stream them one-by-one.

```python
model: str
"""The path to a model file or directory or the name of a SparseZoo model stub."""

model_config: Optional[Dict[str, Any]] = None
"""Keyword arguments passed to the pipeline construction.
Common parameters are sequence_length, prompt_sequence_length"""

generation_config: Union[None, str, Dict] = None
"""GenerationConfig dictionary consisting of parameters used to control
sequences generated for each prompt. Common parameters are:
max_length, max_new_tokens, num_return_sequences, output_scores,
top_p, top_k, repetition_penalty."""

streaming: bool = False
"""Whether to stream the results, token by token."""
```
