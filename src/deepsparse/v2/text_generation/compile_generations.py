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
from typing import Any

import numpy
from pydantic import BaseModel, Field

from deepsparse.transformers.pipelines.text_generation import FinishReason
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import InferenceState


__all__ = ["CompileGenerations", "CompileGenerationsOutput"]


class CompileGenerationsOutput(BaseModel):
    generated_tokens: Any = Field(description="generated_tokens")
    generated_logits: Any = Field(description="generated_logits")
    finished_reason: Any = Field(description="finished_reason")


class CompileGenerations(Operator):
    output_schema = CompileGenerationsOutput

    def can_operate(self, inp: Any):
        if inp.get("in_generation") is False:
            return True
        return False

    def run(self, inference_state: InferenceState, **kwargs):
        generated_tokens = inference_state.current_state.get("generated_tokens")
        generated_logits = inference_state.current_state.get("generated_logits")
        finished_reason = inference_state.current_state.get("finished_reason")

        if len(finished_reason) == 0:
            finished_reason.append(FinishReason.LENGTH)

        generated_tokens = numpy.array([generated_tokens])
        generated_logits = numpy.concatenate(generated_logits, axis=1)
        return {
            "generated_tokens": generated_tokens,
            "generated_logits": generated_logits,
            "finished_reason": finished_reason,
        }
