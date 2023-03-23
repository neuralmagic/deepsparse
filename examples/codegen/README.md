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

## ONNX Export
Firstly, we need to install HuggingFace optimum library
```bash
pip install optimum
```

### Patch the original PyTorch Model
First apply the following modification to this file in your transformers installation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/codegen/modeling_codegen.py#L212

\```diff
-offset = layer_past[0].shape[-2]
+offset = (attention_mask[0] == 0.0).sum() - 1.0
\```

We need to do this because the existing with_past implementations assume there is no padding in the inputs. With deepsparse, we need to use static sequence length, which means our offset for the embeddings will depend on how many non-padded inputs we receive.

The new line checks this with the attention_mask. At this point in the code, attention_mask has been transformed from a tensor with 0s and 1s, to a tensor of `float.min` and `0.0`. So when we compare `attention_mask == 0.0` we are actually saying everywhere the attention_mask is 1.

We also need to subtract 1 from this count, because the attention mask is applied AFTER the kv cache is concatenated to the new token, which means the attention mask will actually be sequence length + 1 items. So we subtract 1 to get the current sequence length.

### Export the model to ONNX

```bash
optimum-cli export onnx --model Salesforce/codegen-350M-multi codegen-350M-multi
```
This saves the model to directory `codegen-350-multi`

### Updating Model's Inputs Outputs Dimension Sizes 
TODO

## Running in the DeepSparse Pipeline

First, we need to rename `decoder_with_past_model.onnx` to `model.onnx` inside
the `static-codegen-350-multi`, to abide the naming convention

Finally, run the pipeline:

```python
from examples.codegen.text_generation import TextGenerationPipeline

codegen = TextGenerationPipeline(
    model_path="/network/damian/static-codegen-350M-multi",
    engine_type="onnxruntime",
    sequence_length=128)

out = codegen(sequences="def hello_world():")
print(out.sequences[0])
```

```bash
def hello_world():
    return 'Hello World!'

def hello_world_2():
    return 'Hello World!'

def hello_world_3():
    return 'Hello World!'

def hello_world_4():
    return 'Hello World!'

def hello_world_5():
    return 'Hello World!'

def hello_world_6():
    return 'Hello World!'

def hello_world_7():
    return 'Hello World!'

def hello_world_8():
    return 'Hello World!'

def hello
```

Modifying pipeline behaviour:
1. By adding argument `deterministic=False`, the next token of the sequence will not be chosen deterministically (using argmax), but will be
sampled from the probablility distribution.
2. By setting `sampling_temperature` when `deterministic=False`, we are allowing more or less randomness in the sampling method (https://towardsdatascience.com/how-to-sample-from-language-models-682bceb97277)
3. By setting `num_tokens_to_generate`, we strictly specify how many tokens we want to generate per input.
