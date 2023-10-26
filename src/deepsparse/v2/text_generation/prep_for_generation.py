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

from deepsparse.transformers.pipelines.text_generation import FinishReason
from deepsparse.v2.operators import Operator
from deepsparse.v2.text_generation import TokenGeneratorOperator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["PrepareGeneration"]


class PrepareGeneration(Operator):
    def __init__(
        self,
        token_generator: TokenGeneratorOperator,
        prompt_sequence_length: int,
        sequence_length: int,
    ):
        self.prompt_sequence_length = prompt_sequence_length
        self.sequence_length = sequence_length
        self.token_generator_creator = token_generator

    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        kv_cache = inp.get("kv_cache")
        tokens = inp.get("tokens")

        # If the number of prompt tokens is greater than what we've processed,
        # don't start generation
        if len(tokens) > kv_cache.total_num_processed_tokens:
            return False
        return True

    @staticmethod
    def set_generated_length(
        max_length: int,
        prompt_tokens_length: int,
        sequence_length: int,
        prompt_sequence_length: int,
        max_new_tokens: int,
        finish_reason_choices: "FinishReason",  # noqa
    ):
        """
        Determine the length of the generated tokens. The hard cap on the total number
        of tokens is based on the sequence length. If max_length is provided and is less
        than the sequence length, it will be used to cap the total number of tokens
        generated. If it is not provided, the max_new_tokens attribute will be used and
        also capped by the sequence length.

        :param max_length: max_length attribute, provided as input during inference
        :param prompt_tokens_length: the number of prompt tokens used as part of the
            generated output
        :param sequence_length: the sequence length used for the pipeline
        :param prompt_sequence_length: the prompt sequence length used for the pipeline
        :param max_new_tokens: the max_new_tokens attribute, which may be provided
        as part of the input during inference
        """
        if max_length:
            # if max_length provided, use that to cap total tokens generated
            max_tokens = max_length
            finish_reason = finish_reason_choices.LENGTH
        else:
            # if not provided, max tokens is based on max_new_tokens + prompt tokens
            max_tokens = (
                min(max_new_tokens, sequence_length - prompt_sequence_length)
                + prompt_tokens_length
            )
            finish_reason = finish_reason_choices.MAX_NEW_TOKENS

        # hard model/pipeline cap
        return (
            (sequence_length, finish_reason_choices.CAPACITY)
            if sequence_length < max_tokens
            else (max_tokens, finish_reason)
        )

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        tokens = inp.get("tokens")
        finished_reason = []
        prompt_logits = inference_state.current_state.get("prompt_logits")
        generation_config = inference_state.current_state.get("generation_config")
        include_prompt_logits = inference_state.current_state.get(
            "include_prompt_logits"
        )

        token_generator, _ = self.token_generator_creator(
            context=context,
            pipeline_state=pipeline_state,
            inference_state=inference_state,
            **{
                "logits_shape": prompt_logits[-1].shape[-1],
                "deterministic": not generation_config.do_sample,
                "sampling_temperature": generation_config.temperature,
                "kwargs": inference_state.current_state,
                "tokens": tokens,
            },
        )
        token_generator = token_generator.get("token_generator")
        token_generator.generate(prompt_logits[0, -1, :])  # TODO: compare/changed
        generated_tokens = [token_generator.tokens[-1]]
        generated_logits = (
            prompt_logits if include_prompt_logits else [prompt_logits[-1]]
        )

        max_tokens, length_finish_reason = PrepareGeneration.set_generated_length(
            max_length=generation_config.max_length,
            prompt_tokens_length=len(generated_tokens),
            max_new_tokens=generation_config.max_new_tokens,
            sequence_length=self.sequence_length,
            prompt_sequence_length=self.prompt_sequence_length,
            finish_reason_choices=FinishReason,
        )
        state_update = {
            "max_tokens": max_tokens,
            "length_finish_reason": length_finish_reason,
            "in_generation": True,
            "generated_tokens": generated_tokens,
            "generated_logits": generated_logits,
            "finished_reason": finished_reason,
            "token_generator": token_generator,
        }

        output = dict(inp)
        output.update({"tokens": token_generator.tokens})
        return output, state_update
