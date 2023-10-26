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

## Using DeepSparse With LangChain

There is a DeepSparse LLM wrapper, which you can access with:

```python
from langchain.llms import DeepSparse
```

It provides a simple, unified interface for all models:

```python
from langchain.llms import DeepSparse
llm = DeepSparse(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none')
print(llm('def fib():'))
```
## Streaming
The DeepSparse LangChain wrapper also supports per token output streaming:

```python
from langchain.llms import DeepSparse
llm = DeepSparse(
    model="hf:neuralmagic/mpt-7b-chat-pruned50-quant",
    streaming=True
)
for chunk in llm.stream("Tell me a joke", stop=["'","\n"]):
    print(chunk, end='', flush=True)
```
## Using Instruction Fine-tune Models With DeepSparse
Here's an example of how to prompt an instruction fine-tuned model using DeepSparse and the MPT-Instruct model:
```python
prompt="""
Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: what is quantization? ### Response:
"""
llm = DeepSparse(model='zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized')
print(llm(prompt))
"""
In physics, the term "quantization" refers to the process of transforming a continuous variable into a set of discrete values. In the context of quantum mechanics, this process is used to describe the restriction of the degrees of freedom of a system to a set of discrete values. In other words, it is the process of transforming the continuous spectrum of a physical quantity into a set of discrete, or "quantized", values.
"""
```
You can also do all the other things you are used to in LangChain such as using `PromptTemplate`s and parsing outputs:
```python
from langchain import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser

llm_parser = CommaSeparatedListOutputParser()
llm = DeepSparse(model='hf:neuralmagic/mpt-7b-chat-pruned50-quant')

prompt = PromptTemplate(
    template="List how to {do}",
    input_variables=["do"])

output = llm.predict(text=prompt.format(do="Become a great software engineer"))

print(output)
"""
List how to Become a great software engineer
By TechRadar Staff
Here are some tips on how to become a great software engineer:
1. Develop good programming skills: To become a great software engineer, you need to have a strong understanding of programming concepts and techniques. You should be able to write clean, efficient code that meets the requirements of the project.
2. Learn new technologies: To stay up-to in the field, you should be familiar with new technologies and programming languages. You should also be able to adapt to new environments and work with different tools and platforms.
3. Build a portfolio: To showcase your skills, you should build a portfolio of your work. This will help you showcase your skills and abilities to potential employers.
4. Network: Networking is an important aspect of your career. You should attend industry events and conferences to meet other professionals in the field.
5. Stay up-to-date with industry trends: Stay up-to-date with industry trends and developments. This will help you stay relevant in your field and help you stay ahead of your competition.
6. Take courses and certifications: Taking courses and certifications can help you gain new skills and knowledge. This will help you stay ahead of your competition and help you grow in your career.
7. Practice and refine your skills: Practice and refine your skills by working on projects and solving problems. This will help you develop your skills and help you grow in your career.
"""

```
## Configuration

The DeepSparse LangChain integration has arguments to control the model loaded, any configs for how the model should be loaded, configs to control how tokens are generated, and then whether to return all tokens at once or to stream them one-by-one.

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
