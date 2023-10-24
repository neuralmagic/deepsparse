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


__all__ = ["AutoRegressiveOperator"]


class AutoRegressiveOperator(Operator):
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        if len(inp.tokens) > self.prompt_sequence_length:
            return False

        start_token = inference_state.current_state.get("start_token")
        end_token = inference_state.current_state.get("end_token")

        if end_token - start_token == 1 and inference_state.current_state.get(
            "batches_processed"
        ) < inference_state.current_state.get("num_batches"):
            return True
        return False

    def _fetch_state_update(self, current: dict):
        return {
            "start_token": current.get("end_token"),
            "end_token": current.get("end_token") + 1,
            "batches_processed": current.get("batches_processed") + 1,
            "num_tokens_processed": current.get("num_tokens_processed") + 1,
        }

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")

        start_token = inference_state.current_state.get("start_token")
        end_token = inference_state.current_state.get("end_token")
        engine_input_names = pipeline_state.current_state.get(
            "onnx_input_names_no_cache"
        )

        new_token = tokens[start_token:end_token]
        num_total_processed_tokens = (
            kv_cache.total_num_processed_tokens
        )  # should be same as the state value?
        print(
            num_total_processed_tokens,
            inference_state.current_state.get("num_total_processed_tokens"),
        )

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

        # filter out the inputs that are not needed by the engine
        engine_inputs_map = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
        )

        # covered by tokens_to_engine_names --> why does this have to be a dictionary? can also remove tokens_to_engins?
        engine_inputs = [engine_inputs_map[name] for name in engine_input_names]

        state_update = self._fetch_state_update(inference_state.current_state)
        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
        }, state_update
