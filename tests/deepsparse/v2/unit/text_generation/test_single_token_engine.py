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

import numpy

from deepsparse.v2.text_generation import (
    AutoRegressiveOperatorPreprocess,
    NLEngineInputs,
)


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


def test_nl_single_token_engine_no_internal(single_token_engine_no_internal_cache):
    assert single_token_engine_no_internal_cache.input_ids_length == 1


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
    inputs = NLEngineInputs(
        engine_inputs=mock_engine_inputs,
        kv_cache=mock_kv_cache_single_token_engine,
        tokens=mock_engine_inputs[0].tolist(),
    )
    output = single_token_engine_no_internal_cache.run(inputs)
    assert output.get("engine_outputs") is not None
