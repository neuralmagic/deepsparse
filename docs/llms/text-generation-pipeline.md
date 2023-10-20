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

# **Text Generation Pipeline**

This user guide explains how to run inference of text generation models with DeepSparse.

## **Installation**

DeepSparse support for LLMs is available on DeepSparse's nightly build on PyPi:

```bash
pip install -U deepsparse-nightly[llm]
```

#### **System Requirements**

- Hardware: x86 AVX2, AVX512, AVX512-VNNI and ARM v8.2+.
- Operating System: Linux (MacOS will be supported soon)
- Python: v3.8-3.11

For those using MacOS or Windows, we suggest using Linux containers with Docker to run DeepSparse.

## **Basic Usage**

DeepSparse exposes a Pipeline interface called `TextGeneration`, which is used to construct a pipeline and generate text:
```python
from deepsparse import TextGeneration

# construct a pipeline
model_path = "zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized"
pipeline = TextGeneration(model=model_path)

# generate text
prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: What is Kubernetes? ### Response:"
output = pipeline(prompt=prompt)
print(output.generations[0].text)

# >> Kubernetes is an open-source container orchestration system for automating deployment, scaling, and management of containerized applications.
```

> **Note:** The 7B model takes about 2 minutes to compile. Set `model_path = hf:mgoin/TinyStories-33M-quant-deepsparse` to use a small TinyStories model for quick compilation if you are just experimenting.

## **Model Format**

DeepSparse accepts models in ONNX format, passed either as SparseZoo stubs or local directories.

> **Note:** DeepSparse uses ONNX graphs modified for KV-caching. We will publish specs to enable external users to create LLM ONNX graphs for DeepSparse over the next few weeks. ***At current, we suggest only using LLM ONNX graphs created by Neural Magic.***
> 
### **SparseZoo Stubs**

SparseZoo stubs identify a model in SparseZoo. For instance, `zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized` identifes a 50% pruned-quantized pretrained MPT-7b model fine-tuned on the Dolly dataset. We can pass the stub to `TextGeneration`, which downloads and caches the ONNX file.

```python
model_path = "zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized"
pipeline = TextGeneration(model=model_path)
```

### **Local Deployment Directory**

Additionally, we can pass a local path to a deployment directory. Use the SparseZoo API to download an example deployment directory:
```python
from sparsezoo import Model
sz_model = Model("zoo:mpt-7b-dolly_mpt_pretrain-pruned50_quantized", "./local-model")
sz_model.deployment.download()
```

Looking at the deployment directory, we see it contains the HF configs and ONNX model files:
```bash
ls ./local-model/deployment
>> config.json model.onnx tokenizer.json model.data special_tokens_map.json tokenizer_config.json
```

We can pass the local directory path to `TextGeneration`:
```python
from deepsparse import TextGeneration
pipeline = TextGeneration(model="./local-model/deployment")
```

### **Hugging Face Models**
Hugging Face models which conform to the directory structure listed above can also be run with DeepSparse by prepending `hf:` to a model id. The following runs a [60% pruned-quantized MPT-7b model trained on GSM](https://huggingface.co/neuralmagic/mpt-7b-gsm8k-pruned60-quant).

```python
from deepsparse import TextGeneration
pipeline = TextGeneration(model="hf:neuralmagic/mpt-7b-gsm8k-pruned60-quant")
```

## **Input and Output Formats**

`TextGeneration` accepts [`TextGenerationInput`](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/pipelines/text_generation.py#L83) as input and returns [`TextGenerationOutput`](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/pipelines/text_generation.py#L170) as output.

The following examples use a quantized 33M parameter TinyStories model for quick compilation:
```python
from deepsparse import TextGeneration

pipeline = TextGeneration(model="hf:mgoin/TinyStories-33M-quant-deepsparse")
```

### Input Format
`TextGenerationInput` has the following fields:
- `sequences` / `prompt`: Input sequences to generate text from. String or list of strings. Required.

```python
prompt1 = "Princess Peach jumped from the balcony"
prompt2 = "Mario ran into the castle"
output = pipeline(sequences=[prompt1, prompt2], max_new_tokens=20)
for prompt_i, generation_i in zip(output.prompts, output.generations):
    print(f"{prompt_i}{generation_i.text}")

# >> Princess Peach jumped from the balcony and landed on the ground. She was so happy that she had found her treasure. She thanked the old

# >> Mario ran into the castle and started to explore. He ran around the castle and climbed on the throne. He even tried to climb
```

- `streaming`: Boolean determining whether to stream response. If True, then the results are returned as a generator object which yields the results as they are generated.

```python
prompt = "Princess Peach jumped from the balcony"
output_iterator = pipeline(prompt=prompt, streaming=True, max_new_tokens=20)

print(prompt, end="")
for it in output_iterator:
    print(it.generations[0].text, end="")

# output is streamed back incrementally
# >> Princess Peach jumped from the balcony and landed on the ground. She was so happy that she had found her treasure. She thanked the old
```

- `generation_config`: Parameters used to control sequences generated for each prompt. [See below for more details](#generation-configuration)
- `generations_kwargs`: Arguments to override the `generation_config` defaults

### Output Format

`TextGenerationOutput` has the following fields:
- `prompts`: String or list of strings. Prompts used for the sequence generation. For multiple input prompts, a list of prompts is returned.
- `generations`: For a single prompt, a list of `GeneratedText` is returned. If multiple prompts are given, a list of `GeneratedText` is returned for each prompt provided. If streaming is enabled, the next generated token is returned. Otherwise, the full generated sequence is returned.
- `created`: Time of inference creation.

`GeneratedText` has the following fields:
- `text`: The generated sequence for a given prompt. If streaming is enabled, this will be the next generated token.
- `score`: The score for the generated token or sequence. The scores have the shape [sequence_length, vocab_size]
- `finished`: Whether generation has stopped.
- `finished_reason`: The reason for generation to stop. Defined by `FinishReason`. One of stop, length, or time.


```python
output = pipeline(sequences=prompt, max_new_tokens=20, output_scores=True)

print(f"created: {output.created}")
print(f"output.prompts: {output.prompts}")
print(f"text: {output.generations[0].text}")
print(f"score.shape: {output.generations[0].score.shape}")
print(f"finished: {output.generations[0].finished}")
print(f"finished_reason: {output.generations[0].finished_reason}")

# >> created: 2023-10-02 13:48:47.660696
# >> prompts: Princess peach jumped from the balcony and
# >> text:  landed on the ground. She was so happy that she had found her treasure. She thanked the bird and
# >> score.shape: (21, 50257)
# >> finished: True
# >> finished_reason: length
```

## **Generation Configuration**

`TextGeneration` can be configured to alter several variables in generation.

The following examples use a quantized 33M parameter TinyStories model for quick compilation:
```python
from deepsparse import TextGeneration

model_id = "hf:mgoin/TinyStories-33M-quant-deepsparse"
pipeline = TextGeneration(model=model_id)
```

### **Creating A `GenerationConfig`**

The `GenerationConfig` can be created in three ways:
- Via `transformers.GenerationConfig`:

```python
from transformers import GenerationConfig

generation_config = GenerationConfig()
generation_config.max_new_tokens = 10
output = pipeline(prompt=prompt, generation_config=generation_config)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

- Via a `dictionary`:
```python
output = pipeline(prompt=prompt, generation_config={"max_new_tokens" : 10})
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

- Via `kwargs`:
```python
output = pipeline(prompt=prompt, max_new_tokens=10)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

### **Passing A `GenerationConfig`**

We can pass a `GenerationConfig` to `TextGeneration.__init__` or `TextGeneration.__call__`.

- If passed to `__init__`, the `GenerationConfig` becomes the default for all subsequent `__call__`s:

```python
# set generation_config during __init__
pipeline_w_gen_config = TextGeneration(model=model_id, generation_config={"max_new_tokens": 10})

# generation_config is the default during __call__
output = pipeline_w_gen_config(prompt=prompt)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

- If passed to `__call__` the `GenerationConfig` will be used for just that generation:

```python
# no generation_config set during __init__
pipeline_w_no_gen_config = TextGeneration(model=model_id)

# generation_config is the passed during __call__
output = pipeline_w_no_gen_config(prompt=prompt, generation_config= {"max_new_tokens": 10})
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

### **Supported `GenerationConfig` Parameters**

The following parameters are supported by the `GenerationConfig`:

#### Controlling The Output
- `output_scores`: Whether to return the generated logits in addition to sampled tokens. Default is `False`

```python
output = pipeline(prompt=prompt, output_scores=True)
print(output.generations[0].score.shape)
# (34, 50257) >> (tokens_generated, vocab_size)
```

- `num_return_sequences`: The number of sequences generated for each prompt. Default is `1`

```python
output = pipeline(prompt=prompt, num_return_sequences=2, do_sample=True, max_new_tokens=10)
for generated_text in output.generations[0]:
    print(f"{prompt}{generated_text.text}")

# >> Princess peach jumped from the balcony and onto her dress. She tried to get away but mummy
# >> Princess peach jumped from the balcony and ran after her. Jill jumped to the floor and followed
```

#### Controling the Output Length
- `max_new_tokens`: maximum number of tokens to generate. Default is `None`
```python
output = pipeline(prompt=prompt, max_new_tokens=10)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she
```

#### Controling the Sampling
- `do_sample`: If True, will apply sampling from the probability distribution computed from the logits rather than deterministic greedy sampling. Default is `False`
```python
output = pipeline(prompt=prompt, do_sample=True, max_new_tokens=15)
print(f"{prompt}{output.generations[0].text}")
output = pipeline(prompt=prompt, do_sample=True, max_new_tokens=15)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and flew down. She used her big teeth to pick it up and gave it some
# >> Princess peach jumped from the balcony and landed in front of her. She stood proudly and exclaimed, “I did
```

- `temperature`: The temperature of the sampling operation. 1 means regular sampling, 0 means always take the highest score, 100.0 is close to uniform probability. If `0.0`, temperature is turned off. Default is `0.0`
```python
# more random
output = pipeline(prompt=prompt, do_sample=True, temperature=1.5, max_new_tokens=15)
print(f"{prompt}{output.generations[0].text}")

# less random
output = pipeline(prompt=prompt, do_sample=True, temperature=0.5, max_new_tokens=15)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and disappeared forever. All that means now is Maria staying where nothing draws herloads.
# >> Princess peach jumped from the balcony and landed on the floor. She was very scared, but she knew that her mom
```
- `top_k`:  Int defining the top tokens considered during sampling. If `0`, `top_k` is turned off. Default is `0`
```python
import numpy

# only 20 logits are not set to -inf == only 20 logits used to sample token
output = pipeline(prompt=prompt, do_sample=True, top_k=20, max_new_tokens=15, output_scores=True)
print(numpy.isfinite(output.generations[0].score).sum(axis=1))
# >> [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
```

- `top_p`: Float to define the tokens that are considered with nucleus sampling. If `0.0`, `top_p` is turned off. Default is `0.0`
```python
import numpy

# small set of logits are not set to -inf == nucleus sampling used 
output = pipeline(prompt=prompt, do_sample=True, top_p=0.9, max_new_tokens=15, output_scores=True)
print(numpy.isfinite(output.generations[0].score).sum(axis=1))

# >> [  5 119  18  14 204   6   7 367 191  20  12   7  46   6   2  35]
```
- `repetition_penalty`: The more a token is used within generation the more it is penalized to not be picked in successive generation passes. If `0.0`, `repetation_penalty` is turned off. Default is `0.0`

```python
output = pipeline(prompt=prompt, repetition_penalty=1.3)
print(f"{prompt}{output.generations[0].text}")
# >> Princess peach jumped from the balcony and landed on the ground. She was so happy that she had found her treasure. She thanked the bird and went back inside to show her family her new treasure.
```
