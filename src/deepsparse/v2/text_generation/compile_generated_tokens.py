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
from typing import Any, Optional

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["CompileGeneratedTokens"]


class CompileGeneratedTokens(Operator):
    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):

        token = inp.get("new_token")
        logits = inp.get("logits")
        finish_reason = inp.get("finish_reason")
        in_generation = True

        generated_tokens = inference_state.current_state.get("generated_tokens")
        generated_logits = inference_state.current_state.get("generated_logits")
        finished_reason = inference_state.current_state.get("finished_reason")

        generated_tokens.append(token)
        generated_logits.append(logits)
        finished_reason.append(finish_reason)

        if finish_reason is not None:
            in_generation = False

        state_update = {  # TODO: check if necessary
            "finished_reason": finished_reason,
            "in_generation": in_generation,
            "generated_tokens": generated_tokens,
            "generated_logits": generated_logits,
        }

        output = {"tokens": inp.get("tokens"), "kv_cache": inp.get("kv_cache")}
        return output, state_update
