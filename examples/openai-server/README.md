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

# OpenAI-compatible Completions Server

Goal: Make a text-generation server that is compatible with the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/introduction) so it can plug-in readily with applications that use the interface.

### Install requirements
`pip install -r requirements.txt`

## Simple CLI usage
Set up the server:
```
python examples/openai-server/server.py --model zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2023-08-07 17:18:32 __main__     INFO     args: Namespace(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none', max_model_len=512, prompt_sequence_length=16, host='localhost', port=8000, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], served_model_name=None)
2023-08-07 17:18:32 deepsparse.transformers WARNING  The neuralmagic fork of transformers may not be installed. It can be installed via `pip install nm_transformers`
Using pad_token, but it is not set yet.
2023-08-07 17:18:34 deepsparse.transformers.engines.nl_decoder_engine INFO     Overwriting in-place the input shapes of the transformer model at /home/mgoin/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/model.onnx
DeepSparse, Copyright 2021-present / Neuralmagic, Inc. version: 1.6.0 COMMUNITY | (98238edf) (optimized) (system=avx512_vnni, binary=avx512)
2023-08-07 17:18:48 deepsparse.transformers.engines.nl_decoder_engine INFO     Overwriting in-place the input shapes of the transformer model at /home/mgoin/.cache/sparsezoo/neuralmagic/codegen_mono-350m-bigpython_bigquery_thepile-base/model.onnx
INFO:     Started server process [314509]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

Query the server for what models are available using the [Models API](https://platform.openai.com/docs/api-reference/models):
```
curl http://localhost:8000/v1/models     
{"object":"list","data":[{"id":"zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none","object":"model","created":1691444523,"owned_by":"neuralmagic","root":"zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none","parent":null,"permission":[{"id":"modelperm-d0d9f0bb6a5c48458848e6b9a8cb8aca","object":"model_permission","created":1691444523,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

Then you can hit the [Completions API](https://platform.openai.com/docs/api-reference/completions) with a `curl` command and see the output:

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        "prompt": "def fib():",
        "max_tokens": 30
    }'

{"id":"cmpl-4d7c32ea65e14468bbe93c63d1687ba9","object":"text_completion","created":1693451394,"model":"zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none","choices":[{"index":0,"text":"\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b","logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"total_tokens":4,"completion_tokens":2}}
```

There is also streaming output to enable with `"stream": true`:

<details>

```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        "prompt": "def fib():",
        "max_tokens": 30,
        "stream": true
    }'

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "def fib():\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "    ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "a, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "b ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "= ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "0, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "1\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "    ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "while ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "True:\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "        ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "yield ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "a\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "        ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "a, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "b ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "= ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "b, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "a ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "+ ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "b", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-14fcb54b0716430bb4f155ffd8882c8f", "object": "text_completion", "created": 1693451416, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": "stop"}]}

data: [DONE]
```
</details>

====

## Code Example

```python
import openai


# Modify OpenAI's API values to use the DeepSparse API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Completion API
stream = False
completion = openai.Completion.create(
    model=model, prompt="def fib():", stream=stream, max_tokens=30
)

print("Completion results:")
if stream:
    text = ""
    for c in completion:
        print(c)
        text += c["choices"][0]["text"]
    print(text)
else:
    print(completion)
```

Output:
<details>

```
Models: {
  "object": "list",
  "data": [
    {
      "id": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
      "object": "model",
      "created": 1693451467,
      "owned_by": "neuralmagic",
      "root": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-611e8298e6974b389e2da6e93b7b576b",
          "object": "model_permission",
          "created": 1693451467,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
Completion results:
{
  "id": "cmpl-caca545954ad4c169b607e36f6a967e4",
  "object": "text_completion",
  "created": 1693451467,
  "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
  "choices": [
    {
      "index": 0,
      "text": "\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 4,
    "completion_tokens": 2
  }
}
```
</details>
