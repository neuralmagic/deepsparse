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

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    :param index: The index of the output in the request.
    :param text: The generated output text.
    :param token_ids: The token IDs of the generated output text.
    :param cumulative_logprob: The cumulative log probability of the generated output
        text.
    :param logprobs: The log probabilities of the top probability words at each
        position if the logprobs are requested.
    :param finish_reason: The reason why the sequence is finished.
    """

    index: int
    text: str
    token_ids: List[int]
    cumulative_logprob: float = 0.0
    logprobs: Optional[List[Dict[int, float]]] = None
    finish_reason: Optional[str] = None

    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class RequestOutput:
    """The output data of a request to the LLM.

    :param request_id: The unique ID of the request.
    :param prompt: The prompt string of the request.
    :param prompt_token_ids: The token IDs of the prompt.
    :param outputs: The output sequences of the request.
    :param finished: Whether the whole request is finished.
    """

    request_id: str
    prompt: str
    prompt_token_ids: List[str]
    outputs: List[CompletionOutput]
    finished: bool
