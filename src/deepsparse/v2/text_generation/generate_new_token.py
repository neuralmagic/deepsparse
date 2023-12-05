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
from typing import Sequence, Union

import transformers

from deepsparse.transformers.pipelines.text_generation import FinishReason
from deepsparse.v2.operators import Operator
from deepsparse.v2.text_generation.nl_engine_operator import NLEngineOutputs
from deepsparse.v2.utils import InferenceState


__all__ = ["GenerateNewTokenOperator"]


class GenerateNewTokenOperator(Operator):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizerBase, force_max_tokens: bool
    ):
        self.force_max_tokens = force_max_tokens
        self.tokenizer = tokenizer

    def can_operate(self, inp: NLEngineOutputs):
        if inp.in_generation:
            return True
        return False

    def run(self, inp: NLEngineOutputs, inference_state: InferenceState, **kwargs):
        logits = inp.engine_outputs
        kv_cache = inp.kv_cache

        token_generator = inference_state.current_state.get("token_generator")
        token = token_generator.generate(logits=logits[0, -1, :])
        finish_reason = None

        callback = inference_state.current_state.get("callback")
        stop = inference_state.current_state.get("stop")

        if token == self.tokenizer.eos_token_id and not self.force_max_tokens:
            finish_reason = FinishReason.STOP

        if self._stop_token_generated(token, stop_tokens=stop):
            print(
                "Stop token %s generated. Stopping generation."
                % self.tokenizer.decode(token)
            )
            finish_reason = FinishReason.STOP

        if callback is not None and callback(token) is False:
            print(
                "callback %s returned False, stopping generation."
                % callback.__qualname__
            )
            finish_reason = FinishReason.CALLBACK

        max_tokens = inference_state.current_state.get("max_tokens")
        if len(inference_state.current_state.get("generated_tokens")) + 1 == max_tokens:
            finish_reason = inference_state.current_state.get("length_finish_reason")

        state_update = {
            "token_generator": token_generator,
        }

        new_generation = {
            "logits": logits,
            "new_token": token,
            "finish_reason": finish_reason,
        }
        output = {"tokens": token_generator.tokens, "kv_cache": kv_cache}
        output.update(new_generation)
        return output, state_update

    def _stop_token_generated(
        self, token, stop_tokens: Union[None, str, Sequence[str]]
    ) -> bool:
        if stop_tokens is None:
            return False

        decoded_token = self.tokenizer.decode(token)
        decoded_token = (
            decoded_token if decoded_token.isspace() else decoded_token.strip()
        )
        return decoded_token in stop_tokens
