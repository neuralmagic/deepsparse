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

from enum import Enum
from typing import Any, Optional

import numpy

from deepsparse.transformers.utils.helpers import create_causal_mask
from deepsparse.v2.operators import Operator
from deepsparse.v2.text_generation.nl_engine_operator import NlEngineInput
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["MultiEnginePrefill"]


class OnnxInputNames(Enum):
    INPUT_IDS = "input_ids"
    ATTN_MASK = "attention_mask"
    CAUSAL_MASK = "causal_mask"
    POSITIONS = "positions"


class MultiEnginePrefill(Operator):
    output_schema = NlEngineInput

    def __init__(self, prompt_sequence_length, sequence_length):
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length
        self.cases = {
            OnnxInputNames.ATTN_MASK.value: self._case_attn_mask,
            OnnxInputNames.POSITIONS.value: self._case_positions,
        }

    # potentially update to use context
    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        if len(inp.tokens) < self.prompt_sequence_length:
            return False

        start_token = inference_state.current_state.get("start_token")
        end_token = inference_state.current_state.get("end_token")

        if (
            end_token - start_token == self.prompt_sequence_length
            and inference_state.current_state.get("batches_processed")
            < inference_state.current_state.get("num_batches")
        ):
            return True
        return False

    def _case_attn_mask(self, num_total_processed_tokens: int):
        # create an empty attention mask
        engine_input = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
        # calculate the number of entries in attention mask that should be set to 1
        num_attention_entries_to_unmask = min(
            num_total_processed_tokens + self.prompt_sequence_length,
            self.sequence_length,
        )
        engine_input[:, -num_attention_entries_to_unmask:] = 1
        return engine_input

    def _case_positions(self, num_total_processed_tokens: int):
        return (
            numpy.arange(
                num_total_processed_tokens,
                num_total_processed_tokens + self.prompt_sequence_length,
            )
            .reshape(1, -1)
            .astype(numpy.int64)
        )

    def _fetch_state_update(self, current: dict):
        return {
            "start_token": current.get("end_token"),
            "end_token": current.get("end_token") + self.prompt_sequence_length,
            "batches_processed": current.get("batches_processed") + 1,
            "num_tokens_processed": current.get("num_tokens_processed")
            + self.prompt_sequence_length,
        }

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        tokens = inp.get("tokens")
        kv_cache = inp.get("kv_cache")
        onnx_input_names_no_cache = pipeline_state.current_state.get(
            "onnx_input_names_no_cache"
        )
        current = inference_state.current_state
        start = current.get("start_multi_token")
        end = current.get("end_multi_token")
        token_batch = tokens[start:end]

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        engine_inputs = []
        for name in onnx_input_names_no_cache:
            if name == OnnxInputNames.INPUT_IDS.value:
                engine_inputs.append(numpy.array([token_batch]))
            elif (
                name == OnnxInputNames.ATTN_MASK.value
                or name == OnnxInputNames.POSITIONS.value
            ):
                engine_inputs.append(self.cases[name](num_total_processed_tokens))

        # create the causal mask once we have the input_ids and attention_mask
        if OnnxInputNames.CAUSAL_MASK.value in onnx_input_names_no_cache:
            causal_mask = create_causal_mask(
                input_ids=engine_inputs[0], attention_mask=engine_inputs[1]
            )
            engine_inputs.append(causal_mask)

        # update state for next token batch to process
        state_update = self._fetch_state_update(current)
        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
        }, state_update
