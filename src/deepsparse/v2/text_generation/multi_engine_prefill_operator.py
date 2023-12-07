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

from deepsparse.transformers.utils.helpers import compute_engine_inputs
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import PipelineState


_LOGGER = logging.getLogger(__name__)

__all__ = ["MultiEnginePrefill"]


class MultiEnginePrefill(Operator):
    def __init__(self, prompt_sequence_length, sequence_length):
        """
        Prepare the tokens for the multi-token engine. This requires creating the
        appropriate engine_inputsto be passed into the multi-token engine.
        """
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length

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

    def run(self, tokens: Any, kv_cache: Any, pipeline_state: PipelineState, **kwargs):
        kv_cache.set_capacity(self.sequence_length - self.prompt_sequence_length)

        num_total_processed_tokens = kv_cache.total_num_processed_tokens
        start = num_total_processed_tokens
        end = start + self.prompt_sequence_length
        token_batch = tokens[start:end]

        engine_inputs = compute_engine_inputs(
            onnx_input_names=pipeline_state.current_state.get(
                "onnx_input_names_no_cache"
            ),
            token_batch=token_batch,
            prompt_sequence_length=self.prompt_sequence_length,
            sequence_length=self.sequence_length,
            num_total_processed_tokens=num_total_processed_tokens,
        )

        return {
            "engine_inputs": engine_inputs,
            "kv_cache": kv_cache,
            "tokens": tokens,
        }
