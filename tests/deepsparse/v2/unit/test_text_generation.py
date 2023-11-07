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

import inspect

import numpy

import pytest
from deepsparse.v2.text_generation import TextGenerationPipeline
from deepsparse.transformers.utils.helpers import prepends_bos_token
from deepsparse.transformers.helpers import get_deployment_path
from transformers import AutoTokenizer
from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.v2.text_generation.process_inputs import GenerationDefaults
from deepsparse.v2.utils import InferenceState
from deepsparse.v2.text_generation import PrepareGeneration, TokenGeneratorOperator, InferenceState
import copy


@pytest.fixture
def text_generation_attributes():
    sequence_length = 5
    prompt_sequence_length = 2
    model_path = "hf:mgoin/TinyStories-1M-deepsparse"
    deployment_path, model_path = get_deployment_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        deployment_path,
        trust_remote_code=False,
        model_max_length=sequence_length,
    )

    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return sequence_length, prompt_sequence_length, model_path, tokenizer


@pytest.fixture
def single_token_engine_no_internal_cache(text_generation_attributes):
    from deepsparse.v2.text_generation import NLEngineOperator
    seq_length, _, model_path, _ = text_generation_attributes
    nl_engine_operator = NLEngineOperator(
        sequence_length=seq_length,
        input_ids_length=1,
        model_path=model_path
    )
    return nl_engine_operator

@pytest.fixture
def pipeline_state(single_token_engine_no_internal_cache):
    from deepsparse.v2.utils import PipelineState

    pipeline_state = PipelineState()
    pipeline_state_vals = {}
    pipeline_state_vals[
        "onnx_input_names_no_cache"
    ] = single_token_engine_no_internal_cache.onnx_input_names_no_cache
    pipeline_state_vals["cache_shape"] = single_token_engine_no_internal_cache.cache_shape
    pipeline_state_vals["output_names"] = single_token_engine_no_internal_cache.output_names
    print(pipeline_state_vals)
    pipeline_state_vals[
        "kv_cache_data_type"
    ] = single_token_engine_no_internal_cache.kv_cache_data_type
    pipeline_state.create_state(pipeline_state_vals)
    return pipeline_state

@pytest.fixture
def large_prompt():
    prompt = "Hello, how are you doing today?"
    generation_config = {"top_p": 0, "top_k": 0, "max_length": 10}
    return TextGenerationInput(prompt=prompt, generation_config=generation_config)

@pytest.fixture
def small_prompt():
    prompt = "Hello"
    return TextGenerationInput(prompt=prompt)

@pytest.fixture
def mock_kv_cache():
    from deepsparse.transformers.utils import DecoderKVCache
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
    )
    return kv_cache

@pytest.fixture
def mock_kv_cache_full():
    from deepsparse.transformers.utils import DecoderKVCache
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
        num_processed_tokens=3
    )
    return kv_cache

"""
@pytest.fixture
def mock_kv_cache_engine(pipeline_state):
    from deepsparse.transformers.utils import DecoderKVCache
    kv_cache = DecoderKVCache()
    kv_cache_state = initialize_kv_cache_state(
        cache_shape=pipeline_state.current_state.get("cache_shape"),
        kv_cache_data_type=pipeline_state.current_state.get("kv_cache_data_type"),
        output_names=pipeline_state.current_state.get("output_names"),
        length=self.sequence_length - self.prompt_sequence_length,
        empty=bool(self.internal_kv_cache),
    )
    print(state)
    return kv_cache
"""

@pytest.fixture
def mock_tokens():
    return [15496]

@pytest.fixture
def mock_tokens_multiple():
    return [15496, 15496, 15496]

@pytest.fixture
def mock_inference_state():
    generation_config = GenerationDefaults()
    inference_state = InferenceState()
    inference_state.create_state({})
    inference_state.update_state({
        "generation_config": generation_config})
    return inference_state

@pytest.fixture
def mock_token_generator(text_generation_attributes, mock_tokens_multiple):
    _, _, _, tokenizer = text_generation_attributes
    token_generator_creator = TokenGeneratorOperator()
    prompt_logits = numpy.random.rand(1, len(mock_tokens_multiple), len(tokenizer))
    token_generator_creator_output = token_generator_creator.run(
        logits_shape=prompt_logits[0, -1, :].shape,
        deterministic=True,
        sampling_temperature=1.0,
        tokens=copy.copy(mock_tokens_multiple),
    )
    return token_generator_creator_output.get("token_generator")

@pytest.fixture
def mock_logits(text_generation_attributes):
    _, _, _, tokenizer = text_generation_attributes
    return numpy.random.rand(1, 1, len(tokenizer))


def test_process_inputs(text_generation_attributes, small_prompt, large_prompt):
    sequence_length, _, _, tokenizer = text_generation_attributes
    from deepsparse.v2.text_generation.process_inputs import ProcessInputsTextGeneration
    process_inputs = ProcessInputsTextGeneration(
        sequence_length=sequence_length,
        tokenizer=tokenizer
    )

    outputs, state_update = process_inputs.run(small_prompt)
    assert len(outputs.get("tokens")) == 1
    assert isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get("prompts") == small_prompt.sequences

    outputs, state_update = process_inputs.run(large_prompt)
    
    assert not isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get("generation_config").max_length == large_prompt.generation_config.get("max_length")
    assert outputs.get("tokens")
    assert state_update.get("top_k") == large_prompt.generation_config.get("top_k")


def test_nl_single_token_engine_no_internal(single_token_engine_no_internal_cache):
    assert single_token_engine_no_internal_cache.input_ids_length == 1
    
def test_kv_cache_creation(pipeline_state, text_generation_attributes):
    from deepsparse.v2.text_generation import KVCacheCreator, KVCacheCreatorInput
    seq_length, prompt_sequence_length, model_path, tokenizer = text_generation_attributes
    kv_cache_creator = KVCacheCreator(
        tokenizer=tokenizer,
        prompt_sequence_length=prompt_sequence_length,
        sequence_length=seq_length,
        internal_kv_cache=False
    )
    
    assert kv_cache_creator.input_schema == KVCacheCreatorInput
    kv_cache = kv_cache_creator.run(
        cache_shape=pipeline_state.current_state.get("cache_shape"),
        kv_cache_data_type=pipeline_state.current_state.get("kv_cache_data_type"),
        output_names=pipeline_state.current_state.get("output_names")
    )
    assert kv_cache.get("kv_cache")
    assert kv_cache.get("kv_cache").total_num_processed_tokens == 0


def test_autoreg_preproces_can_run(text_generation_attributes, pipeline_state, mock_tokens, mock_kv_cache):
    seq_len, prompt_seq_len,  _, _ = text_generation_attributes
    from deepsparse.v2.text_generation.autoregressive_preprocess_operator import AutoRegressiveOperatorPreprocess
    autoreg_prep = AutoRegressiveOperatorPreprocess(
        sequence_length=seq_len,
        prompt_sequence_length=prompt_seq_len
    )
    inputs = {"tokens": mock_tokens, "kv_cache": mock_kv_cache}

    assert autoreg_prep.can_operate(inputs)
    outputs = autoreg_prep.run(
        tokens=mock_tokens,
        kv_cache=mock_kv_cache,
        pipeline_state=pipeline_state
    )

    assert len(outputs.get("engine_inputs")) == 4 # tokens, attention mask, causal, positions
    tokens, attention_mask, positions, causal_mask = outputs.get("engine_inputs")
    print(outputs.get("engine_inputs"))
    assert tokens.shape[-1] == 1
    assert attention_mask.shape[-1] == seq_len
    assert positions[0] == mock_kv_cache.total_num_processed_tokens
    assert outputs.get("in_generation") is None

def test_autoreg_preproces_cant_run(text_generation_attributes, mock_kv_cache, mock_tokens_multiple):
    seq_len, _, _, _ = text_generation_attributes
    from deepsparse.v2.text_generation.autoregressive_preprocess_operator import AutoRegressiveOperatorPreprocess
    autoreg_prep = AutoRegressiveOperatorPreprocess(
        sequence_length=seq_len,
        prompt_sequence_length=2
    )
    inputs = {"tokens": mock_tokens_multiple, "kv_cache": mock_kv_cache}
    assert not autoreg_prep.can_operate(inputs)
    
def test_mult_engine_preprocess(text_generation_attributes, mock_kv_cache, mock_tokens_multiple, pipeline_state):
    seq_len, prompt_seq_len, _, _ = text_generation_attributes
    from deepsparse.v2.text_generation.multi_engine_prefill_operator import MultiEnginePrefill
    multi_prep = MultiEnginePrefill(
        sequence_length=seq_len,
        prompt_sequence_length=prompt_seq_len
    )
    inputs = {"tokens": mock_tokens_multiple, "kv_cache": mock_kv_cache}
    assert multi_prep.can_operate(inputs)
    outputs = multi_prep.run(tokens=mock_tokens_multiple, kv_cache=mock_kv_cache, pipeline_state=pipeline_state)    
    assert len(outputs.get("engine_inputs")) == 4 # tokens, attention mask, causal, positions
    tokens, attention_mask, positions, causal_mask = outputs.get("engine_inputs")
    assert tokens.shape[-1] == prompt_seq_len
    assert attention_mask.shape[-1] == seq_len
    assert positions.shape[-1] == prompt_seq_len

def test_multi_engine_preprocess_cant_operate(text_generation_attributes, mock_kv_cache, mock_tokens):
    seq_len, prompt_seq_len, _, _ = text_generation_attributes
    from deepsparse.v2.text_generation.multi_engine_prefill_operator import MultiEnginePrefill
    multi_prep = MultiEnginePrefill(
        sequence_length=seq_len,
        prompt_sequence_length=prompt_seq_len
    )
    inputs = {"tokens": mock_tokens, "kv_cache": mock_kv_cache}
    assert not multi_prep.can_operate(inputs)

"""
def test_run_single_engine_once(single_token_engine_no_internal_cache, mock_kv_cache_engine):
    from deepsparse.v2.text_generation.nl_engine_operator import NlEngineInput

    mock_engine_inputs = [numpy.array([[15496]]), numpy.array([[0, 0, 0, 0, 1]]), numpy.array([[0]]), numpy.array([[[[0, 0, 0, 0, 1]]]])]
    inputs = NlEngineInput(
        engine_inputs=mock_engine_inputs,
        kv_cache=mock_kv_cache_engine,
        tokens=mock_engine_inputs[0].tolist()
    )
    print(single_token_engine_no_internal_cache.run(inputs))
"""

def test_prep_for_generation(mock_tokens_multiple, mock_kv_cache_full, text_generation_attributes, mock_inference_state):
    seq_len, prompt_seq_len, _, tokenizer = text_generation_attributes
    prep_for_generation = PrepareGeneration(
        token_generator=TokenGeneratorOperator(),
        sequence_length=seq_len,
        prompt_sequence_length=prompt_seq_len
    )
    inputs = {"tokens": mock_tokens_multiple, "kv_cache": mock_kv_cache_full}
    assert prep_for_generation.can_operate(inputs)

    prompt_logits = [numpy.random.rand(1, len(mock_tokens_multiple), len(tokenizer))]
    mock_inference_state.update_state({"prompt_logits": prompt_logits})
    outputs, state = prep_for_generation.run(
        tokens=mock_tokens_multiple,
        kv_cache=mock_kv_cache,
        inference_state=mock_inference_state
    ) 
    assert len(outputs.get("tokens")) == len(mock_tokens_multiple) + 1
    assert outputs.get("in_generation")
    assert numpy.array_equal(state.get("generated_logits")[0], numpy.expand_dims(prompt_logits[0][:, -1, :], 0))

def test_generate_new_token(mock_token_generator, text_generation_attributes, mock_kv_cache, mock_inference_state, mock_logits, mock_tokens):
    _, _, _, tokenizer = text_generation_attributes
    from deepsparse.v2.text_generation import GenerateNewTokenOperator
    generate_new_token = GenerateNewTokenOperator(
        force_max_tokens=False,
        tokenizer=tokenizer
    )
    mock_inference_state.update_state({"token_generator": mock_token_generator, "generated_tokens": [mock_token_generator.tokens]})
    outputs, state = generate_new_token.run(
        logits=mock_logits,
        kv_cache=mock_kv_cache,
        inference_state=mock_inference_state
    )
    assert outputs.get("new_token") == state.get("token_generator").tokens[-1]


def test_compile_logits(mock_logits, mock_inference_state):
    from deepsparse.v2.text_generation import CompilePromptLogits
    mock_inference_state.update_state({"prompt_logits": [mock_logits]})
    compile_prompt_logits = CompilePromptLogits()
    assert compile_prompt_logits.can_operate({})
    output, state = compile_prompt_logits.run(
        logits=mock_logits,
        inference_state=mock_inference_state
    )
    assert len(state.get("prompt_logits")) == len([mock_logits]) + 1