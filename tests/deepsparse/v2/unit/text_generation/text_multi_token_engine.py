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

from deepsparse.v2.text_generation import MultiEnginePrefill


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
