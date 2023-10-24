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

from pydantic import Field

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["PrepareforSingleEngine"]


class PrepareforSingleEngine(Operator):
    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        number_tokens_processed = inference_state.current_state.get(
            "num_tokens_processed"
        )
        tokens = inp.get("tokens")
        if (
            len(tokens) < self.prompt_sequence_length
        ):  ## can't run multi-engine (running first time)
            return True

        for c in context.stages_executed:
            if c.operator.__name__ == "AutoRegressiveOperator":
                return False

        if (
            len(tokens[number_tokens_processed:]) == 0
        ):  ## if 0 remain, can't operate (after multi-engine has already run)
            return False
        return True  ## if some remain, can operate

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        tokens = inp.get("tokens")
        num_processed_tokens = inference_state.current_stat.get(
            "num_tokens_processed", 0
        )
        state_dict = {
            "num_batches": len(tokens[num_processed_tokens:]),
            "start_token": num_processed_tokens,
            "end_token": num_processed_tokens + 1,
            "batches_processed": 0,
            "num_processed_tokens": num_processed_tokens,
        }
        return inp, state_dict
