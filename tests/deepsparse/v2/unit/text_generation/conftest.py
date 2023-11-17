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
from transformers import AutoTokenizer

import pytest
from deepsparse.transformers.helpers import get_deployment_path
from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.transformers.utils import DecoderKVCache
from deepsparse.transformers.utils.helpers import initialize_kv_cache_state
from deepsparse.v2 import InferenceState, PipelineState
from deepsparse.v2.text_generation import (
    GenerationDefaults,
    NLEngineOperator,
    TokenGeneratorOperator,
)


@pytest.fixture(scope="module")
def text_generation_attributes():
    sequence_length = 5
    prompt_sequence_length = 1
    return sequence_length, prompt_sequence_length


@pytest.fixture(scope="module")
def model_attributes(text_generation_attributes):
    model_path = "hf:mgoin/TinyStories-1M-deepsparse"
    sequence_length, _ = text_generation_attributes
    deployment_path, model_path = get_deployment_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        deployment_path,
        trust_remote_code=False,
        model_max_length=sequence_length,
    )

    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model_path


@pytest.fixture(scope="module")
def single_token_engine_no_internal_cache(text_generation_attributes, model_attributes):
    seq_length, _ = text_generation_attributes
    _, model_path = model_attributes

    nl_engine_operator = NLEngineOperator(
        sequence_length=seq_length, input_ids_length=1, model_path=model_path
    )
    return nl_engine_operator


@pytest.fixture(scope="module")
def pipeline_state(single_token_engine_no_internal_cache):
    pipeline_state = PipelineState()
    pipeline_state_vals = {}
    pipeline_state_vals[
        "onnx_input_names_no_cache"
    ] = single_token_engine_no_internal_cache.onnx_input_names_no_cache
    pipeline_state_vals[
        "cache_shape"
    ] = single_token_engine_no_internal_cache.cache_shape
    pipeline_state_vals[
        "output_names"
    ] = single_token_engine_no_internal_cache.output_names
    pipeline_state_vals[
        "kv_cache_data_type"
    ] = single_token_engine_no_internal_cache.kv_cache_data_type
    pipeline_state.create_state(pipeline_state_vals)
    return pipeline_state


@pytest.fixture(scope="module")
def large_prompt():
    prompt = "Hello, how are you doing today?"
    generation_config = {"top_p": 0, "top_k": 0, "max_length": 10}
    return TextGenerationInput(prompt=prompt, generation_config=generation_config)


@pytest.fixture(scope="module")
def small_prompt():
    prompt = "Hello"
    return TextGenerationInput(prompt=prompt)


@pytest.fixture(scope="module")
def mock_kv_cache():
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
    )
    return kv_cache


@pytest.fixture(scope="module")
def mock_kv_cache_three_tokens_processed():
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
        num_processed_tokens=3,
    )
    return kv_cache


@pytest.fixture(scope="module")
def mock_kv_cache_single_token_engine(pipeline_state, text_generation_attributes):
    seq_len, prompt_seq_len = text_generation_attributes
    kv_cache = DecoderKVCache()
    kv_cache_state = initialize_kv_cache_state(
        cache_shape=pipeline_state.current_state.get("cache_shape"),
        kv_cache_data_type=pipeline_state.current_state.get("kv_cache_data_type"),
        output_names=pipeline_state.current_state.get("output_names"),
        length=seq_len - prompt_seq_len,
        empty=False,
    )
    kv_cache.setup(state=kv_cache_state)
    return kv_cache


@pytest.fixture(scope="module")
def mock_tokens():
    return [15496]


@pytest.fixture(scope="module")
def mock_tokens_multiple():
    return [15496, 15496, 15496]


@pytest.fixture(scope="module")
def mock_inference_state():
    generation_config = GenerationDefaults()
    inference_state = InferenceState()
    inference_state.create_state({})
    inference_state.update_state({"generation_config": generation_config})
    return inference_state


@pytest.fixture(scope="module")
def mock_token_generator(model_attributes, mock_tokens_multiple):
    tokenizer, _ = model_attributes
    token_generator_creator = TokenGeneratorOperator()
    prompt_logits = numpy.random.rand(1, len(mock_tokens_multiple), len(tokenizer))
    token_generator_creator_output = token_generator_creator.run(
        logits_shape=prompt_logits[0, -1, :].shape,
        deterministic=True,
        sampling_temperature=1.0,
        tokens=copy.copy(mock_tokens_multiple),
    )
    return token_generator_creator_output.get("token_generator")


@pytest.fixture(scope="module")
def mock_logits(model_attributes):
    tokenizer, _ = model_attributes
    return numpy.random.rand(1, 1, len(tokenizer))
