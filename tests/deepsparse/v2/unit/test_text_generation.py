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
    AutoRegressiveOperatorPreprocess,
    CompilePromptLogits,
    GenerateNewTokenOperator,
    GenerationDefaults,
    KVCacheCreator,
    KVCacheCreatorInput,
    MultiEnginePrefill,
    NlEngineInput,
    NLEngineOperator,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    TokenGeneratorOperator,
)


@pytest.fixture
def text_generation_attributes():
    sequence_length = 5
    prompt_sequence_length = 1
    return sequence_length, prompt_sequence_length


@pytest.fixture
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


@pytest.fixture
def single_token_engine_no_internal_cache(text_generation_attributes, model_attributes):
    seq_length, _ = text_generation_attributes
    _, model_path = model_attributes

    nl_engine_operator = NLEngineOperator(
        sequence_length=seq_length, input_ids_length=1, model_path=model_path
    )
    return nl_engine_operator


@pytest.fixture
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
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
    )
    return kv_cache


@pytest.fixture
def mock_kv_cache_three_tokens_processed():
    kv_cache = DecoderKVCache()
    kv_cache.setup(
        state={"dummy_cache_name": numpy.array([[[[0], [0], [1], [2], [3]]]])},
        num_processed_tokens=3,
    )
    return kv_cache


@pytest.fixture
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
    inference_state.update_state({"generation_config": generation_config})
    return inference_state


@pytest.fixture
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


@pytest.fixture
def mock_logits(model_attributes):
    tokenizer, _ = model_attributes
    return numpy.random.rand(1, 1, len(tokenizer))


def test_process_inputs(
    text_generation_attributes, model_attributes, small_prompt, large_prompt
):
    """
    Check if the ProcessInputsTextGeneration Operator successfully processes the
    inputs and generation config.
    """
    sequence_length, _ = text_generation_attributes
    tokenizer, _ = model_attributes
    process_inputs = ProcessInputsTextGeneration(
        sequence_length=sequence_length, tokenizer=tokenizer
    )

    outputs, state_update = process_inputs.run(small_prompt)
    assert len(outputs.get("tokens")) == 1
    assert isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get("prompts") == small_prompt.sequences

    outputs, state_update = process_inputs.run(large_prompt)

    assert not isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get(
        "generation_config"
    ).max_length == large_prompt.generation_config.get("max_length")
    assert outputs.get("tokens")
    assert state_update.get("top_k") == large_prompt.generation_config.get("top_k")


def test_nl_single_token_engine_no_internal(single_token_engine_no_internal_cache):
    assert single_token_engine_no_internal_cache.input_ids_length == 1


def test_kv_cache_creation(
    text_generation_attributes, model_attributes, pipeline_state
):
    """
    Check if the KVCacheCreator successfully creates a kv_cache object, given the
    single_token_engine attributes stored in the pipeline_state.
    """
    seq_length, prompt_seq_len = text_generation_attributes
    tokenizer, _ = model_attributes
    kv_cache_creator = KVCacheCreator(
        tokenizer=tokenizer,
        prompt_sequence_length=prompt_seq_len,
        sequence_length=seq_length,
        internal_kv_cache=False,
    )

    assert kv_cache_creator.input_schema == KVCacheCreatorInput
    kv_cache = kv_cache_creator.run(
        cache_shape=pipeline_state.current_state.get("cache_shape"),
        kv_cache_data_type=pipeline_state.current_state.get("kv_cache_data_type"),
        output_names=pipeline_state.current_state.get("output_names"),
    )
    assert kv_cache.get("kv_cache")
    assert kv_cache.get("kv_cache").total_num_processed_tokens == 0


def test_autoreg_preproces_can_run(
    text_generation_attributes, pipeline_state, mock_tokens, mock_kv_cache
):
    """
    Check if the single-token engine preprocess operator can run based on the provided
    tokens and prompt_sequence_length.
    """

    seq_len, _ = text_generation_attributes
    autoreg_prep = AutoRegressiveOperatorPreprocess(
        sequence_length=seq_len, prompt_sequence_length=len(mock_tokens) + 1
    )
    inputs = {"tokens": mock_tokens, "kv_cache": mock_kv_cache}

    # The prompt_sequence_length is greater than the number of tokens that are to be
    # operated on. Therefore, use the single_token_engine and can_operate() should be
    # True.
    assert autoreg_prep.can_operate(inputs)
    outputs = autoreg_prep.run(
        tokens=mock_tokens, kv_cache=mock_kv_cache, pipeline_state=pipeline_state
    )
    # Assert 4 engine inputs: tokens, attention mask, causal, positions
    assert len(outputs.get("engine_inputs")) == 4
    tokens, attention_mask, positions, causal_mask = outputs.get("engine_inputs")

    assert tokens.shape[-1] == 1
    assert attention_mask.shape[-1] == seq_len
    assert positions[0] == mock_kv_cache.total_num_processed_tokens
    assert outputs.get("in_generation") is None


def test_autoreg_preproces_cant_run(
    text_generation_attributes, mock_kv_cache, mock_tokens_multiple
):
    """
    Check if the single-token engine preprocess operator can run based on the provided
    tokens and prompt_sequence_length.
    """

    seq_len, _ = text_generation_attributes
    autoreg_prep = AutoRegressiveOperatorPreprocess(
        sequence_length=seq_len, prompt_sequence_length=len(mock_tokens_multiple)
    )
    inputs = {"tokens": mock_tokens_multiple, "kv_cache": mock_kv_cache}
    # can_operate() should be False as the prompt_sequence_length is equal to the
    # number of tokens we want to operate on. Therefore, the multi-token engine
    # should run instead.
    assert not autoreg_prep.can_operate(inputs)


def test_mult_engine_preprocess(
    text_generation_attributes, pipeline_state, mock_kv_cache, mock_tokens_multiple
):
    """
    Check if the multi-token engine preprocess operator can run based on the provided
    tokens and prompt_sequence_length.
    """

    seq_len, _ = text_generation_attributes
    multi_prep = MultiEnginePrefill(
        sequence_length=seq_len, prompt_sequence_length=len(mock_tokens_multiple)
    )
    inputs = {"tokens": mock_tokens_multiple, "kv_cache": mock_kv_cache}
    # The number of tokens is equal to the prompt_sequence_length.
    # Therefore, the multi_token_engine can run and can_operate() should be True.
    assert multi_prep.can_operate(inputs)
    outputs = multi_prep.run(
        tokens=mock_tokens_multiple,
        kv_cache=mock_kv_cache,
        pipeline_state=pipeline_state,
    )
    # Expect 4 engine inputs: tokens, attention mask, causal, positions
    assert len(outputs.get("engine_inputs")) == 4
    tokens, attention_mask, positions, causal_mask = outputs.get("engine_inputs")
    # Assert proper shapes for all engine_inputs
    assert tokens.shape[-1] == len(mock_tokens_multiple)
    assert attention_mask.shape[-1] == seq_len
    assert positions.shape[-1] == len(mock_tokens_multiple)


def test_multi_engine_preprocess_cant_operate(
    text_generation_attributes, mock_kv_cache, mock_tokens
):
    """
    Check if the multi-token engine preprocess operator can run based on the provided
    tokens and prompt_sequence_length.
    """
    seq_len, _ = text_generation_attributes
    multi_prep = MultiEnginePrefill(
        sequence_length=seq_len, prompt_sequence_length=len(mock_tokens) + 1
    )
    inputs = {"tokens": mock_tokens, "kv_cache": mock_kv_cache}
    # The prompt_sequence_length is one greater than the total number of tokens we're
    # processing. Therefore, this operator should not run and can_operate() should be
    # False.
    assert not multi_prep.can_operate(inputs)


def test_run_single_token_engine_once(
    single_token_engine_no_internal_cache,
    mock_kv_cache_single_token_engine,
):
    """
    This operator runs through the single-token NLEngine once, given engine_inputs and
    kv_cache.
    """

    mock_engine_inputs = [
        numpy.array([[15496]]),
        numpy.array([[0, 0, 0, 0, 1]]),
        numpy.array([[0]]),
        numpy.array([[[[0, 0, 0, 0, 1]]]]),
    ]
    inputs = NlEngineInput(
        engine_inputs=mock_engine_inputs,
        kv_cache=mock_kv_cache_single_token_engine,
        tokens=mock_engine_inputs[0].tolist(),
    )
    output = single_token_engine_no_internal_cache.run(inputs)
    assert output.get("logits") is not None


def test_prep_for_generation(
    text_generation_attributes,
    model_attributes,
    mock_tokens_multiple,
    mock_kv_cache_three_tokens_processed,
    mock_inference_state,
):
    """
    This test will assess the PrepareGeneration, which runs after prompt_inference
    and before generation.
    """
    seq_len, prompt_seq_len = text_generation_attributes
    tokenizer, _ = model_attributes
    prep_for_generation = PrepareGeneration(
        prompt_sequence_length=prompt_seq_len,
        token_generator=TokenGeneratorOperator(),
        sequence_length=seq_len,
    )
    inputs = {
        "tokens": mock_tokens_multiple,
        "kv_cache": mock_kv_cache_three_tokens_processed,
    }
    # can_operate() if the total number of prompt tokens is equal to the
    # number of processed tokens stored in the kv_cache, indicating prompt inference is
    # complete and generation can begin.
    assert prep_for_generation.can_operate(inputs)

    prompt_logits = [numpy.random.rand(1, len(mock_tokens_multiple), len(tokenizer))]
    mock_inference_state.update_state({"prompt_logits": prompt_logits})
    outputs, state = prep_for_generation.run(
        tokens=mock_tokens_multiple,
        kv_cache=mock_kv_cache,
        inference_state=mock_inference_state,
    )
    assert len(outputs.get("tokens")) == len(mock_tokens_multiple) + 1
    assert outputs.get("in_generation")
    assert numpy.array_equal(
        state.get("generated_logits")[0],
        numpy.expand_dims(prompt_logits[0][:, -1, :], 0),
    )


def test_generate_new_token(
    model_attributes,
    mock_token_generator,
    mock_kv_cache,
    mock_inference_state,
    mock_logits,
):
    """
    This test is responsible for testing the GenerateNewTokenOperator, which generates
    one new token, given a token_generator (stored in the inference_state) and logits
    from the engine.
    """
    tokenizer, _ = model_attributes
    generate_new_token = GenerateNewTokenOperator(
        force_max_tokens=False, tokenizer=tokenizer
    )
    mock_inference_state.update_state(
        {
            "token_generator": mock_token_generator,
            "generated_tokens": [mock_token_generator.tokens],
        }
    )
    outputs, state = generate_new_token.run(
        logits=mock_logits, kv_cache=mock_kv_cache, inference_state=mock_inference_state
    )
    # The new_token generated/returned by ths operator should match the last token in
    # token_generator
    assert outputs.get("new_token") == state.get("token_generator").tokens[-1]


def test_compile_logits(mock_logits, mock_inference_state):
    mock_inference_state.update_state({"prompt_logits": [mock_logits]})
    compile_prompt_logits = CompilePromptLogits()
    # Can operate as long as we're not in generation but in prompt_inference. This
    # can_operate() will check for the `in_generation` flag in the input.
    assert compile_prompt_logits.can_operate({})
    output, state = compile_prompt_logits.run(
        logits=mock_logits, inference_state=mock_inference_state
    )
    # The CompilePromptLogits is responsible for updating a list of prompt logits
    # calculated at each step during prompt inference. After one step of running this
    # operator, the total number of prompt_logits in the inference state should be
    # the current length of prompt logits + 1
    assert len(state.get("prompt_logits")) == len([mock_logits]) + 1
