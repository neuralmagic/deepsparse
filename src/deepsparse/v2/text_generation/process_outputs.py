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
from typing import Optional

import numpy

from deepsparse.transformers.pipelines.text_generation import (
    FinishReason,
    GeneratedText,
    TextGenerationOutput,
)
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import InferenceState


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
        generated_tokens: numpy.ndarray,
        generated_logits: numpy.ndarray,
        finished_reason: list,
        inference_state: InferenceState,
        **kwargs,
    ):
        generation_config = inference_state.current_state.get("generation_config")
        generated_logits = generated_logits if generation_config.output_scores else None
        sequences = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        finished_reason = [f[-1] for f in finished_reason]

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

        num_preds = generation_config.num_return_sequences
        if num_preds > 1:
            grouped_generations = [
                generations[n : n + num_preds]
                for n in range(0, len(generations), num_preds)
            ]
            generations = grouped_generations

        outputs = dict(
            created=datetime.datetime.now(),
            prompts=inference_state.current_state.get("prompts"),
            generations=generations,
        )

        return outputs
