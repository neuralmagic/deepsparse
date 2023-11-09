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

from deepsparse.v2.text_generation import CompilePromptLogits


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
