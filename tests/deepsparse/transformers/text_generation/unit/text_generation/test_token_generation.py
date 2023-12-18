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

from deepsparse.transformers.pipelines.text_generation import (
    GenerateNewTokenOperator,
    PrepareGeneration,
    TokenGeneratorOperator,
)
from deepsparse.transformers.pipelines.text_generation.nl_engine_operator import (
    NLEngineOutputs,
)


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
        kv_cache=mock_kv_cache_three_tokens_processed,
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
    mock_tokens,
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
    inp = NLEngineOutputs(
        engine_outputs=mock_logits,
        tokens=mock_tokens,
        kv_cache=mock_kv_cache,
        in_generation=True,
    )
    outputs, state = generate_new_token.run(
        inp=inp, inference_state=mock_inference_state
    )
    # The new_token generated/returned by ths operator should match the last token in
    # token_generator
    assert outputs.get("new_token") == state.get("token_generator").tokens[-1]
