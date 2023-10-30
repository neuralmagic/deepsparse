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

import pathlib
from typing import Any, Dict, Optional, Union

import transformers

from deepsparse.transformers.pipelines.text_generation import TextGenerationInput
from deepsparse.transformers.utils.helpers import (
    check_and_return_generation_config,
    override_config,
    repeat_inputs,
)
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


class GenerationDefaults:
    num_return_sequences = 1
    max_length = 10
    max_new_tokens = None
    output_scores = False
    top_k = 0
    top_p = 0.0
    repetition_penalty = 0.0
    do_sample = False
    temperature = 1.0


__all__ = ["ProcessInputsTextGeneration"]


class ProcessInputsTextGeneration(Operator):
    """
    Input processing operator. Responsible for tokenizing the input, handling the
    generation_config (if provided), updating the inference_state for later use,
    and returning the tokens for prompt inferece. The expected input is defined by
    the input_schema, which for this operator is TextGeneratioInput.
    """

    input_schema = TextGenerationInput

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        sequence_length: int,
        generation_config: Union[
            str, pathlib.Path, Dict, transformers.GenerationConfig, None
        ] = None,
    ):
        self.generation_config = generation_config or {}
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        generation_config = check_and_return_generation_config(
            self.generation_config, inp.generation_config, GenerationDefaults()
        )

        generation_config = override_config(inp.generation_kwargs, generation_config)

        original_inputs = inp.sequences
        if generation_config.num_return_sequences > 1:
            if isinstance(inp.sequences, str):
                inp.sequences = [inp.sequences]
            inp.sequences = repeat_inputs(
                inp.sequences, generation_config.num_return_sequences
            )

        if inp.fixed_sequences_length:
            # to enforce a fixed sequence length, we need to
            # truncate the input to the maximum sequence length
            # or/and pad it to the maximum sequence length
            truncate, padding = True, "max_length"
        else:
            # otherwise, we do not need to truncate the input
            # and we shall can pad it to the longest sequence
            # in the batch (so that the engine can process multiple inputs
            # at once)
            truncate, padding = False, "longest"

        input_tokens = self.tokenizer(
            inp.sequences,
            return_tensors="np",
            max_length=self.sequence_length,
            padding=padding,
            truncation=truncate,
        )

        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens["attention_mask"]

        inference_state_update = dict(
            prompts=original_inputs,
            streaming=inp.streaming,
            generation_config=generation_config,
            include_prompt_logits=inp.include_prompt_logits,
            callback=inp.callback,
            stop=inp.stop,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            presence_penalty=inp.presence_penalty,
            frequency_penalty=generation_config.repetition_penalty,
        )

        # TODO: move this step to prep_for_prefill and add attention mask to the output
        # this will allow us to split/join more easily when processing multiple prompts
        # in parallel
        tokens = input_ids[attention_mask.nonzero()].tolist()

        return {
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, inference_state_update
