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

from deepsparse.v2.text_generation import (
    GenerationDefaults,
    ProcessInputsTextGeneration,
)


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
    assert len(outputs.get("input_ids")) == 1
    assert len(outputs.get("attention_mask")) == 1
    assert isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get("prompts") == small_prompt.sequences

    outputs, state_update = process_inputs.run(large_prompt)

    assert not isinstance(state_update.get("generation_config"), GenerationDefaults)
    assert state_update.get(
        "generation_config"
    ).max_length == large_prompt.generation_config.get("max_length")
    assert outputs.get("input_ids") is not None
    assert state_update.get("top_k") == large_prompt.generation_config.get("top_k")
