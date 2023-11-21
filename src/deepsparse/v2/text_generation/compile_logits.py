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

from deepsparse.v2.operators import Operator
from deepsparse.v2.text_generation.nl_engine_operator import NLEngineOutputs
from deepsparse.v2.utils import InferenceState


__all__ = ["CompilePromptLogits"]


class CompilePromptLogits(Operator):
    """
    Combine the prompt logits. Currently relying on the inference state to store the
    prompt logits for each token or multi-token batch processed. This operator will
    take prompt logits from each iteration run and update the inference state.
    """

    def can_operate(self, inp: NLEngineOutputs):
        if inp.in_generation is None:
            return True
        return False

    def run(self, inp: NLEngineOutputs, inference_state: InferenceState, **kwargs):
        logits = inp.engine_outputs
        logit_type = "prompt_logits"

        if inference_state.current_state.get(logit_type) is not None:
            current_logits = inference_state.current_state.get(logit_type).copy()
            current_logits.append(logits)
        else:
            current_logits = [logits]

        state_update = {logit_type: current_logits}
        return {
            "kv_cache": inp.kv_cache,
            "tokens": inp.tokens,
        }, state_update
