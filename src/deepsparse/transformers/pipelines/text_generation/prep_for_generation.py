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
import copy
from typing import Any, Optional

import numpy

from deepsparse.operators import Operator
from deepsparse.subgraph_execute import StreamingOutput
from deepsparse.transformers.pipelines.text_generation import TokenGeneratorOperator
from deepsparse.transformers.schemas.text_generation_schemas import (
    FinishReason,
    PromptLogitsNoKVCacheInference,
)
from deepsparse.transformers.utils.helpers import set_generated_length
from deepsparse.utils import InferenceState


__all__ = ["PrepareGeneration"]


class PrepareGeneration(Operator):
    def __init__(
        self,
        token_generator: TokenGeneratorOperator,
        prompt_sequence_length: int,
        sequence_length: int,
        process_output_operator: Optional[Operator] = None,
    ):
        self.sequence_length = sequence_length
        self.token_generator_creator = token_generator
        self.prompt_sequence_length = prompt_sequence_length
        # Needed for streaming as currently both setting up generation and generating
        # Will split this up soon
        self.process_output_operator = process_output_operator

    def can_operate(self, inp: Any):
        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")

        # If the number of prompt tokens is greater
        # than what we've processed, don't start generation.
        # Should be equal when started as all prompt logits
        # should be accounted for, and we should have updated
        # the kv_cache for the single token engine.
        if len(tokens) == kv_cache.total_num_processed_tokens:
            return True
        return False

    def run(
        self, tokens: Any, kv_cache: Any, inference_state: InferenceState, **kwargs
    ):
        prompt_logits = inference_state.current_state.get("prompt_logits")
        prompt_logits = numpy.concatenate(prompt_logits, axis=1)
        # TODO: clean this up such that dont have to keep writing current_state
        # everywhere

        generation_config = inference_state.current_state.get("generation_config")
        include_prompt_logits = inference_state.current_state.get(
            "include_prompt_logits"
        )

        token_generator_creator_output = self.token_generator_creator.run(
            logits_shape=prompt_logits[0, -1, :].shape,
            deterministic=not generation_config.do_sample,
            sampling_temperature=generation_config.temperature,
            tokens=copy.copy(tokens),
            **inference_state.current_state,
        )
        token_generator = token_generator_creator_output.get("token_generator")
        token_generator.generate(prompt_logits[0, -1, :])

        max_tokens, length_finish_reason = set_generated_length(
            max_length=generation_config.max_length,
            prompt_tokens_length=1,
            max_new_tokens=generation_config.max_new_tokens,
            sequence_length=self.sequence_length,
            prompt_sequence_length=self.prompt_sequence_length,
            finish_reason_choices=FinishReason,
        )

        state_update = {
            "max_tokens": max_tokens,
            "length_finish_reason": length_finish_reason,
            "generated_tokens": [token_generator.tokens[-1]],
            "generated_logits": [prompt_logits]
            if include_prompt_logits
            else [numpy.expand_dims(prompt_logits[:, -1, :], 0)],
            "finished_reason": [],
            "token_generator": token_generator,
        }
        if kv_cache is None:
            output = PromptLogitsNoKVCacheInference(prompt_logits=prompt_logits)
        else:
            output = {
                "tokens": token_generator.tokens,
                "kv_cache": kv_cache,
                "in_generation": True,
            }
            # TODO: maybe break this operator up since it is both generating and setting
            # up values needed for generation? Holding off on this as this will change
            # routes slighty and want to confirm wont break anything for non-kv cache
            if inference_state.current_state.get("streaming") and max_tokens >= 1:
                finished_reason = [length_finish_reason] if max_tokens == 1 else [None]

                if self.process_output_operator is None:
                    raise ValueError(
                        "An operator must be provided to process outputs"
                        "while streaming."
                    )
                data_to_yield = self.process_output_operator.run(
                    generated_tokens=state_update.get("generated_tokens"),
                    finished_reason=finished_reason,
                    inference_state=inference_state,
                    generated_logits=prompt_logits[0, -1, :],
                )
                output = StreamingOutput(
                    data_to_yield=self.process_output_operator.output_schema(
                        **data_to_yield
                    ),
                    data_to_return=output,
                )

        return output, state_update
