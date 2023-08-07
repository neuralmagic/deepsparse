Goal: Make a text-generation server that is compatible with the [OpenAI API Reference](https://platform.openai.com/docs/api-reference/introduction) so it can plug-in readily with applications that use the interface.

## Install requirements
`pip install deepsparse-nightly[server] transformers`

## Example usage
Set up the server:
```
python examples/openai-server/server.py --model zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none --prompt-processing-sequence-length 1
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2023-08-07 17:18:32 __main__     INFO     args: Namespace(model='zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none', max_model_len=512, prompt_processing_sequence_length=1, use_deepsparse_cache=False, host='localhost', port=8000, allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], served_model_name=None)
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

Then you can hit the [Completions API](https://platform.openai.com/docs/api-reference/completions) with a `curl` command and see the streaming output:
```
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none",
        "prompt": "def fib():",
        "max_tokens": 16,
        "stream": true
    }'
data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "def fib():\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "a, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "0, ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "  ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "while ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "True:\n", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "  ", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: {"id": "cmpl-473d4978ecc64a61a5eb6c442505aeba", "object": "text_completion", "created": 1691444444, "model": "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/bigpython_bigquery_thepile/base-none", "choices": [{"index": 0, "text": "", "logprobs": null, "finish_reason": null}]}

data: [DONE]
```

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
stream = True
completion = openai.Completion.create(
    model=model,
    prompt="def fib():",
    stream=stream,
    max_tokenss=32)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
```

Output:
