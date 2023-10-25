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

import numpy

from deepsparse.transformers.utils.helpers import create_causal_mask
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["AutoRegressiveOperatorPreprocess"]


class AutoRegressiveOperatorPreprocess(Operator):
    def __init__(self, sequence_length: int, prompt_sequence_length: int):
        """
        Prepare the tokens for the single-token engine. This requires creating the
        attention mask, positions, and causal mask. The output contains these three
        arrays to be passed into the single-token engine.
        """
        self.sequence_length = sequence_length
        self.prompt_sequence_length = prompt_sequence_length

    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        """
        Can run this Operator if the number of tokens left to process is greater than
        0 but less than the self.promt_sequence_length. Also, thie Operator can only
        run after PrepareforSingleEngine as it requires the kv_cache to be updated.
        """
        tokens = inp.get("tokens")
        kv_cache = inp.get("kv_cache")

        found = False
        for c in context.stages_executed:
            if c.operator.__class__.__name__ == "PrepareforSingleEngine":
                found = True

        remaining_tokens = len(tokens) - kv_cache.total_num_processed_tokens
        if found and (
            remaining_tokens > 0 and remaining_tokens < self.prompt_sequence_length
        ):
            return True
        return False

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        new_token = tokens[num_total_processed_tokens]

        # padding is added to left, so attention mask is 1s from the
        # right up to the number of total tokens (prompt + generated)
        attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        num_attention_entries_to_unmask = min(
            num_total_processed_tokens + 1, self.sequence_length
        )  # cap by seq len
        attention_mask[:, -num_attention_entries_to_unmask:] = 1
        positions = numpy.array([[num_total_processed_tokens]], dtype=numpy.int64)
        input_ids = numpy.array([[new_token]])
        causal_mask = create_causal_mask(input_ids, attention_mask)

        engine_inputs_map = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
        )

        onnx_input_names_no_cache = pipeline_state.current_state.get(
            "onnx_input_names_no_cache"
        )
        engine_inputs = [engine_inputs_map[name] for name in onnx_input_names_no_cache]

        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
        }, {}
