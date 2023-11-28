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
from typing import Dict, Union

import transformers

from deepsparse.transformers.pipelines.text_generation import (
    GenerationDefaults,
    TextGenerationInput,
)
from deepsparse.transformers.utils.helpers import (
    check_and_return_generation_config,
    override_config,
    repeat_inputs,
)
from deepsparse.v2.operators import Operator


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
            str, pathlib.Path, Dict, transformers.GenerationConfig
        ] = None,
    ):
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def run(self, inp: TextGenerationInput, **kwargs):
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, inference_state_update
