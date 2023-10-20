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

import numpy as np

import pytest
from deepsparse import Pipeline


@pytest.fixture
def pipeline():
    return Pipeline.create(
        task="chat",
        model_path="hf:mgoin/TinyStories-1M-deepsparse",
        engine_type="onnxruntime",
    )


@pytest.fixture
def prompt():
    return "Never gonna give you up, never gonna let you down"


def test_chat_pipeline_session_manager(pipeline, prompt):
    with pipeline.session():
        output_1 = pipeline(prompt)
        output_2 = pipeline(prompt)
    # assert inferences in the same context share a session id
    assert output_1.session_ids == output_2.session_ids

    # test that follow-up inference has a different session id
    output_3 = pipeline(prompt)
    assert output_3.session_ids != output_1.session_ids


def test_run_with_same_session_ids(pipeline):
    # Test the scenario where the same session ids are used for multiple
    # inference runs. There are two conditions that must be fulfilled:
    # 1. The information regarding the prompt does not leak between sessions
    # 2. Running two prompts one after another is identical to running
    #    a composition of those prompts i.e.
    #     generated_text = pipeline(prompt_1)
    #     generated_text_2 = pipeline(prompt_2)
    #     generated_text_2 == pipeline(prompt_1 + generated_text + prompt_2)

    prompt_1 = "This prompt is used for testing purposes. To this to make sure that"
    prompt_2 = "still this prompt should not"

    # make sure information does not leak between sessions
    _test_composition_same_session_ids(
        prompt_1=prompt_1,
        prompt_2=prompt_2,
        num_generated_tokens=32,
        pipeline=pipeline,
        session_id_1="test_1",
        session_id_2="test_2",
    )

    _test_composition_same_session_ids(
        prompt_1=prompt_1,
        prompt_2=prompt_2,
        num_generated_tokens=32,
        pipeline=pipeline,
        session_id_1="test_3",
        session_id_2="test_4",
    )


def _test_composition_same_session_ids(
    prompt_1,
    prompt_2,
    num_generated_tokens,
    pipeline,
    session_id_1,
    session_id_2,
):

    tokenizer = pipeline.tokenizer

    # make sure that running two prompts one after another
    # is identical to running a composition of those prompts
    out_1_ = pipeline(
        sequences=prompt_1,
        force_max_tokens=True,
        session_ids=session_id_1,
        max_new_tokens=num_generated_tokens,
        include_prompt_logits=True,
    )
    prompt_1_ = out_1_.generations[0].text
    out_1 = pipeline(
        sequences=prompt_2,
        force_max_tokens=True,
        session_ids=session_id_1,
        max_new_tokens=num_generated_tokens,
        include_prompt_logits=True,
    )
    cache_state_1 = pipeline.storage_kv_cache.get(session_id_1).cached_inputs[
        "past_key_values.0.key"
    ]

    prompt_composition = tokenizer.decode(
        tokenizer(prompt_1).input_ids
        + tokenizer(prompt_1_).input_ids
        + tokenizer(prompt_2).input_ids,
        skip_special_tokens=True,
    )
    out_2 = pipeline(
        sequences=prompt_composition,
        max_new_tokens=num_generated_tokens,
        session_ids=session_id_2,
        include_prompt_logits=True,
    )
    cache_state_2 = pipeline.storage_kv_cache.get(session_id_2).cached_inputs[
        "past_key_values.0.key"
    ]
    if cache_state_1.shape[0]:
        # if cache state is not empty, i.e. we are managing kv cache
        # externally, make sure that the cache state is the same
        np.allclose(cache_state_1, cache_state_2, atol=0.001)
    assert out_1.generations[0].text == out_2.generations[0].text
