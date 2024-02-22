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

import numpy

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
    max_length = 16
    pipeline = TextGeneration(
        model_path=model_attributes[1], onnx_model_name="model-orig.onnx"
    )
    non_kv_cache_logits = []
    for _ in range(max_length):
        # simulate autoregressive generation with non-kv cache pipeline
        out = pipeline(prompt=_prompt, output_scores=True)
        non_kv_cache_logits.append(out.generations[0].score)
        new_token = pipeline.tokenizer.encode(out.generations[0].text)
        old_tokens = pipeline.tokenizer.encode(_prompt)
        _prompt = pipeline.tokenizer.decode(old_tokens + new_token)

    pipeline_kv_cache = TextGeneration(model_path=model_attributes[1])
    out = pipeline_kv_cache(prompt=prompt, max_length=max_length, output_scores=True)
    kv_cache_scores = out.generations[0].score

    non_kv_cache_logits = numpy.concatenate(non_kv_cache_logits, axis=0)

    assert numpy.allclose(non_kv_cache_logits, kv_cache_scores, atol=0.001)
    assert _prompt == prompt + out.generations[0].text
