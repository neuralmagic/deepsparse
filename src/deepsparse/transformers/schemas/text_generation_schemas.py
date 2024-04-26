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

import datetime
import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from transformers import GenerationConfig


# Based off of https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig # noqa E501
@dataclass
class GenerationDefaults:
    # Parameters that control the length of the output
    max_length: int = field(default=None)
    max_new_tokens: int = field(default=100)
    # Parameters that control the generation strategy used
    do_sample: bool = field(default=False)
    # Parameters for manipulation of the model output logits
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    repetition_penalty: float = field(default=1.0)
    # Parameters that define the outputs
    num_return_sequences: int = field(default=1)
    output_scores: bool = field(default=False)


class FinishReason(Enum):
    STOP = "stop"
    LENGTH = "length"
    TIME = "time"
    CALLBACK = "callback"
    CAPACITY = "capacity"
    MAX_NEW_TOKENS = "max_new_tokens"


class TextGenerationInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequences: Union[str, List[str]] = Field(
        alias="prompt",
        description="The input sequences to generate the text from.",
    )
    return_input_tokens: bool = Field(
        default=False,
        description="A flag that indicates whether to return the input_tokens. ",
    )
    include_prompt_logits: bool = Field(
        default=False,
        description="A flag that indicates whether to return "
        "the logits for the prompt. If set, prompt_logits are "
        "`prepended` to the logits for the generated text sequence."
        "Note: This flag is only applicable when output_scores "
        "is `True`.",
    )
    fixed_sequences_length: bool = Field(
        default=False,
        description="A flag that indicates whether to modify "
        "(pad or truncate) each input text sequence, so that "
        "its tokenized length is equal to `sequence_length` "
        "of tokens. Useful, when a batch of predictions needs "
        "to have consistent length so one "
        "can compute metric in a batched fashion. ",
    )
    streaming: bool = Field(
        default=False,
        description="Whether to stream the results back as they are generated. If "
        "True, then the results are returned as a generator object which yields "
        "the results as they are generated. If False, then the results are returned "
        "as a list after it has completed.",
    )
    callback: Optional[Callable[[Any], Union[bool, Any]]] = Field(
        default=None,
        description="Callable that will be invoked "
        "on each generated token. If the callable returns "
        "`False`, the generation will stop. Default is `None`.",
    )
    stop: Union[None, str, Sequence[str]] = Field(
        default=None,
        description="A string or a list of strings that will be used as"
        " stop tokens. (token generation will stop when any of the stop"
        " tokens is generated). Set to `None` to ignore this parameter."
        " Default is `None`.",
    )

    presence_penalty: Optional[float] = Field(
        default=0.0,
        description="Penalty applied for generating new token. Any existing"
        " token results in the subtraction of its corresponding logit value."
        " Default set to 0.0",
    )

    generation_config: Union[None, str, pathlib.Path, Dict, GenerationConfig] = Field(
        default=None,
        description="GenerationConfig file consisting of parameters used to control "
        "sequences generated for each prompt. The current supported parameters are: "
        "max_length, max_new_tokens, num_return_sequences, output_scores, top_p, "
        "top_k, repetition_penalty, do_sample, temperature. If None is provided, "
        "deepsparse defaults will be used. For all other input types, HuggingFace "
        "defaults for GenerationConfig will be used. ",
    )

    generation_kwargs: Optional[Dict] = Field(
        default=None,
        description="Any arguments to override generation_config arguments. Refer to "
        "the generation_config argument for a full list of supported variables.",
    )


class GeneratedText(BaseModel):
    text: str = Field(
        description="The generated sequence for a given prompt. If "
        "streaming is enabled, this will be the next generated token."
    )
    score: Optional[Any] = Field(
        default=None,
        description="The score for the generated token or sequence. "
        "The scores have the shape [sequence_length, vocab_size]",
    )
    finished: bool = Field(description="Whether generation has stopped.")
    finished_reason: Optional[str] = Field(
        default=None,
        description="The reason for generation to stop. "
        "Defined by FinishReason. One of stop, length, or time.",
    )


# TODO: Pydantic aliases allow assignment but not reference. Still need to update.
class TextGenerationOutput(BaseModel):
    created: datetime.datetime = Field(description="Time of inference creation.")
    prompts: Union[str, List[str]] = Field(
        description="Prompts used for the sequence generation. For multiple input "
        "prompts, a list of prompts is returned"
    )
    generations: Union[List[GeneratedText], List[List[GeneratedText]]] = Field(
        description="For a single prompt, a single list of GeneratedText is returned. "
        "If multiple prompts are given, a list of GeneratedText is returned for each "
        "prompt provided. If streaming is enabled, the next generated token is "
        "returned. Otherwise, the full generated sequence is returned."
    )
    input_tokens: Optional[
        Any
    ] = Field(  # dictionary mapping "token_ids" and "attention_mask" to numpy arrays
        default=None,
        description="The output of the tokenizer."
        "Dictionary containing token_ids and attention_mask, "
        "both mapping to arrays of size [batch_size, sequence_length]",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
