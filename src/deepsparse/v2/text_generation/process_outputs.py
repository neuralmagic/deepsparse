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
from typing import Any, Optional

import numpy

from deepsparse.transformers.pipelines.text_generation import (
    FinishReason,
    GeneratedText,
    TextGenerationOutput,
)
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


class ProcessOutputs(Operator):
    output_schema = TextGenerationOutput

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _create_generated_text_output(
        self,
        sequence: str,
        finish_reason: Optional[FinishReason] = None,
        logits: Optional[numpy.array] = None,
    ):
        if finish_reason:
            return GeneratedText(
                text=sequence,
                score=logits,
                finished=True,
                finished_reason=finish_reason.value,
            )
        return GeneratedText(
            text=sequence,
            score=logits,
            finished=False,
        )

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        generation_config = inference_state.current_state.get("generation_config")
        generated_tokens = inp["tokens"]  # inp.generated_tokens
        generated_logits = (
            inp["generated_logits"] if generation_config.output_scores else None
        )
        finished_reason = inp["finished_reason"]  # inp.finished_reason
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        finished_reason = [f for f in finished_reason if f]

        if generated_logits is not None:
            generations = list(
                map(
                    self._create_generated_text_output,
                    sequences,
                    finished_reason,
                    generated_logits,
                )
            )
        else:
            generations = list(
                map(self._create_generated_text_output, sequences, finished_reason)
            )
        outputs = dict(
            created=datetime.datetime.now(),
            prompts=inference_state.current_state.get("prompts"),
            generations=generations,
        )

        return outputs, {}
