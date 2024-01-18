# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from deepsparse import TextGeneration
from deepsparse.transformers.pipelines.text_generation.pipeline import (
    TextGenerationPipeline,
)
from deepsparse.transformers.pipelines.text_generation.pipeline_no_kv_cache import (
    TextGenerationPipelineNoCache,
)


def test_create_pipeline_with_kv_cache_support(model_attributes):
    pipeline = TextGeneration(model_path=model_attributes[1])
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = TextGeneration(model_path=model_attributes[1], onnx_model_name=None)
    assert isinstance(pipeline, TextGenerationPipeline)


def test_create_pipeline_with_no_kv_cache_support(model_attributes):
    pipeline = TextGeneration(
        model_path=model_attributes[1], onnx_model_name="model-orig.onnx"
    )
    assert isinstance(pipeline, TextGenerationPipelineNoCache)


def test_assert_same_outputs_regardless_of_kv_cache_support(model_attributes):
    # make sure that kv cache support does not change the output
    prompt = "Hello, how are you doing today?"
    _prompt = copy.deepcopy(prompt)
    max_new_tokens = 16
    pipeline = TextGeneration(
        model_path=model_attributes[1], onnx_model_name="model-orig.onnx"
    )
    for _ in range(max_new_tokens):
        # simulate autoregressive generation with non-kv cache pipeline
        out = pipeline(prompt=_prompt)
        new_token = pipeline.tokenizer.encode(out.generations[0].text)
        old_tokens = pipeline.tokenizer.encode(_prompt)
        _prompt = pipeline.tokenizer.decode(old_tokens + new_token)
    # max_new_tokens reduced by one, because the pipeline always grabs
    # the first generated token from the prefill
    out = TextGeneration(model_path=model_attributes[1])(
        prompt=prompt, max_new_tokens=max_new_tokens - 1
    )
    assert _prompt == prompt + out.generations[0].text
