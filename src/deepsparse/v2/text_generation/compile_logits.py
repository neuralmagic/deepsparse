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

import numpy as np

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["CompilePromptLogits"]


class CompilePromptLogits(Operator):
    """
    Combine the prompt logits. Currently relying on the inference state to store the
    prompt logits for each token or multi-token batch processed. This operator will
    take prompt logits from each iteration run and update the inference state.
    """

    def can_operate(self, inp: Any, context: Context):
        if inp.get("in_generation"):
            return False

        found = False
        for c in context.stages_executed:
            if c.operator.__class__.__name__ == "PrepareGeneration":
                return True

        if not found:
            return True
        return False

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        logit_type = "prompt_logits"
        logits = inp.get("logits")

        if inference_state.current_state.get(logit_type) is not None:
            current_logits = inference_state.current_state.get(logit_type).copy()
            current_logits = np.concatenate((current_logits, logits), axis=1)
        else:
            current_logits = logits

        state_update = {logit_type: current_logits}
        return inp, state_update
