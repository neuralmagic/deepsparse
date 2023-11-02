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
from enum import Enum
from typing import Any

import numpy

from deepsparse.transformers.utils.helpers import create_causal_mask
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import PipelineState


_LOGGER = logging.getLogger(__name__)

__all__ = ["MultiEnginePrefill"]


class OnnxInputNames(Enum):
    INPUT_IDS = "input_ids"
    ATTN_MASK = "attention_mask"
    CAUSAL_MASK = "causal_mask"
    POSITIONS = "positions"


# NOTE: A possible clean-up could involve combining this Operator and the
# autoregressive_preprocess_operator


class MultiEnginePrefill(Operator):
    def __init__(self, prompt_sequence_length, sequence_length):
        """
        Prepare the tokens for the multi-token engine. This requires creating the
        attention mask, positions, and causal mask. The output contains these three
        arrays to be passed into the multi-token engine.
        """
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length
        self.cases = {
            OnnxInputNames.ATTN_MASK.value: self._case_attn_mask,
            OnnxInputNames.POSITIONS.value: self._case_positions,
        }
        _LOGGER.warn(
            "This operator requires the PipelineState to be set-up with the "
            "onnx_input_names_no_cache attribute set from the NLEngineOperator."
        )

    def can_operate(self, inp: Any):
        """
        Can only run if the number of prompt tokens left to process is greater than
        or equal to the self.prompt_sequence_length.
        """
        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")

        if len(tokens) < self.prompt_sequence_length:
            return False

        if (
            len(tokens) - kv_cache.total_num_processed_tokens
            >= self.prompt_sequence_length
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

    def run(self, tokens: Any, kv_cache: Any, pipeline_state: PipelineState, **kwargs):
        kv_cache.set_capacity(self.sequence_length - self.prompt_sequence_length)
        onnx_input_names_no_cache = pipeline_state.current_state.get(
            "onnx_input_names_no_cache"
        )

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        start = num_total_processed_tokens
        end = start + self.prompt_sequence_length
        token_batch = tokens[start:end]

        engine_inputs = []
        for name in onnx_input_names_no_cache:
            if name == OnnxInputNames.INPUT_IDS.value:
                engine_input = numpy.array([token_batch])
            elif (
                name == OnnxInputNames.ATTN_MASK.value
                or name == OnnxInputNames.POSITIONS.value
            ):
                engine_input = self.cases[name](num_total_processed_tokens)
            elif name == OnnxInputNames.CAUSAL_MASK.value:
                continue

            engine_inputs.append(engine_input)

        if OnnxInputNames.CAUSAL_MASK.value in onnx_input_names_no_cache:
            causal_mask = create_causal_mask(
                input_ids=engine_inputs[0],
                attention_mask=engine_inputs[1],
            )
            engine_inputs.append(causal_mask)

        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
        }
