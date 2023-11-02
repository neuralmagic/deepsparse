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

import logging
from typing import Any

import numpy

from deepsparse.transformers.utils.helpers import create_causal_mask
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import PipelineState


_LOGGER = logging.getLogger(__name__)

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
        self.set_capacity = False

        _LOGGER.warn(
            "This operator requires the PipelineState to be set-up with the "
            "onnx_input_names_no_cache attribute set from the NLEngineOperator."
        )

    def can_operate(self, inp: Any) -> bool:
        """
        Can run this Operator if the number of tokens left to process is greater than
        0 but less than the self.prompt_sequence_length.
        """
        tokens = inp.get("tokens")
        kv_cache = inp.get("kv_cache")

        if inp.get("in_generation"):
            return True

        remaining_tokens = len(tokens) - kv_cache.total_num_processed_tokens
        can_process = (
            remaining_tokens > 0 and remaining_tokens < self.prompt_sequence_length
        )
        if can_process and inp.get("in_generation") is None:
            return True
        return False

    def run(self, tokens: Any, kv_cache: Any, pipeline_state: PipelineState, **kwargs):
        if not self.set_capacity or (
            self.set_capacity and kwargs.get("in_generation") is None
        ):
            self.set_capacity = True
            kv_cache.set_capacity(self.sequence_length - 1)

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        new_token = tokens[num_total_processed_tokens]
        engine_input_names = pipeline_state.current_state.get(
            "onnx_input_names_no_cache"
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

        engine_inputs_map = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            positions=positions,
        )

        engine_inputs = [engine_inputs_map[name] for name in engine_input_names]

        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
            "in_generation": kwargs.get("in_generation"),
        }
