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
from typing import Any, Optional, Sequence, Union

import numpy
import transformers

from deepsparse.transformers.pipelines.text_generation import FinishReason
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


class GenerateNewTokenOperator(Operator):
    def __init__(
        self, tokenizer: transformers.PreTrainedTokenizerBase, force_max_tokens: bool
    ):
        self.force_max_tokens = force_max_tokens
        self.tokenizer = tokenizer

    def can_operate(self, inp: Any, context: Context, inference_state: InferenceState):
        if inference_state.current_state.get("in_generation"):
            return True
        return False

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

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        logits = inp.get("logits")
        in_generation = True
        token_generator = inference_state.current_state.get("token_generator")
        token = token_generator.generate(logits=logits[0, -1, :])

        generated_tokens = inference_state.current_state.get("generated_tokens")
        generated_logits = inference_state.current_state.get("generated_logits")
        finished_reason = inference_state.current_state.get("finished_reason")
        max_tokens = inference_state.current_state.get("max_tokens")
        callback = inference_state.current_state.get("callback")
        stop = inference_state.current_state.get("stop")

        generated_tokens.append(token)
        generated_logits.append(logits)

        if token == self.tokenizer.eos_token_id and not self.force_max_tokens:
            finished_reason.append(FinishReason.STOP)
            in_generation = False

        if self._stop_token_generated(token, stop_tokens=stop):
            print(
                "Stop token %s generated. Stopping generation."
                % self.tokenizer.decode(token)
            )
            finished_reason.append(FinishReason.STOP)
            in_generation = False

        if callback is not None and callback(token) is False:
            print(
                "callback %s returned False, stopping generation."
                % callback.__qualname__
            )
            finished_reason.append(FinishReason.CALLBACK)
            in_generation = False

        if len(inference_state.current_state.get("generated_tokens")) == max_tokens:
            finished_reason.append(
                inference_state.current_state.get("length_finish_reason")
            )
            in_generation = False

        if in_generation is False:
            if len(finished_reason) == 0:
                finished_reason.append(FinishReason.LENGTH)

            generated_tokens = numpy.array([generated_tokens])
            generated_logits[0] = numpy.expand_dims(generated_logits[0], axis=0)
            generated_logits = numpy.concatenate(generated_logits, axis=1)

        state_update = {  # TODO: check if necessary
            "finished_reason": finished_reason,
            "in_generation": in_generation,
            "generated_tokens": generated_tokens,
            "generated_logits": generated_logits,
            "token_generator": token_generator,
        }
        output = dict(inp)
        output.update({"tokens": token_generator.tokens})
        return output, state_update
