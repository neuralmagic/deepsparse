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

```bash
pip install optimum
optimum-cli export onnx --model Salesforce/codegen-350M-multi codegen-350M-multi
```
This saves the model to directory `codegen-350-multi`

## Adapting the ONNX model
To run the model in the pipeline, we need to slightly adjust the model:

### Updating Model's Inputs Outputs Dimension Sizes 
TODO

### Fixing the 'offset variable'
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
