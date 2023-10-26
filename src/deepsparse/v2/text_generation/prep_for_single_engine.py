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


__all__ = ["PrepareforSingleEngine"]


class PrepareforSingleEngine(Operator):
    def __init__(self, prompt_sequence_length: int, sequence_length: int):
        """
        Prepare to use the single_engine Operator for prompt inference. This requires
        updating the kv_cache capacity.
        """
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length

    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        # Don't rerun if in autoregessive loop
        for c in context.stages_executed:
            if c.operator.__class__.__name__ == "AutoRegressiveOperatorPreprocess":
                return False

        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")
        # if number of prompt tokens left to process is >= self.prompt_sequnce_length
        # should use the multi_token engine.
        if (
            len(tokens) - kv_cache.total_num_processed_tokens
            >= self.prompt_sequence_length
        ):
            return False

        # if 0 prompt tokens remain, can't operate (multi-token engine has already run)
        # if len(tokens) == kv_cache.total_num_processed_tokens:
        #    return False

        return True

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        kv_cache = inp.get("kv_cache")
        kv_cache.set_capacity(self.sequence_length - 1)

        input_values = dict(inp)
        input_values.update({"kv_cache": kv_cache})
        return input_values, {}
