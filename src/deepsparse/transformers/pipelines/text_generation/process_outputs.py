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
from typing import List, Optional

import numpy

from deepsparse.operators import Operator
from deepsparse.subgraph_execute import StreamingOutput
from deepsparse.transformers.schemas.text_generation_schemas import (
    FinishReason,
    GeneratedText,
    TextGenerationOutput,
)
from deepsparse.utils import InferenceState


__all__ = ["ProcessOutputs", "ProcessStreamingOperator"]


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

    def _generate_streamed_text_from_past_tokens(
        self, generated_tokens: numpy.ndarray, past_tokens_queue: List[int]
    ) -> str:
        """
        An auxiliary method that helps to properly generate the streamed text.
        Some models like llama2 and mistral are using LlamaTokenizer which is
        based on SentencePiece tokenizer. This specific tokenizer doesn't seem
        to output appropriate prefix spaces when decoding token by token.
        One can make it work if the previously generated tokens are included.
        This allows the tokenizer to figure out that the appropriate spaces
        from last n consecutive tokens.

        :param generated_tokens: the generated tokens from the engine
        :param past_tokens_queue: the queue of last n tokens (n is the
            original prompt length in tokens)
        :return: the generated string
        """
        string_from_n_tokens = self.tokenizer.decode(
            past_tokens_queue, skip_special_tokens=True
        )
        past_tokens_queue.append(generated_tokens[0])
        string_from_n_plus_1_tokens = self.tokenizer.decode(
            past_tokens_queue, skip_special_tokens=True
        )
        past_tokens_queue.pop(0)
        return [string_from_n_plus_1_tokens[len(string_from_n_tokens) :]]

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

        import transformers

        # Fix for LLAMA-specific models when running streaming
        # TODO: make streaming a conditional input to this operator. using inference
        # state is a quick fix.
        if isinstance(
            self.tokenizer,
            (transformers.LlamaTokenizer, transformers.LlamaTokenizerFast),
        ) and inference_state.current_state.get("streaming"):
            past_tokens_queue = inference_state.current_state.get("past_tokens_queue")
            sequences = self._generate_streamed_text_from_past_tokens(
                generated_tokens, past_tokens_queue
            )
        else:
            sequences = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )

        try:
            finished_reason = [f[-1] for f in finished_reason]
        except Exception:
            pass

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
        if num_preds > 1 and not inference_state.current_state.get("streaming"):
            grouped_generations = [
                generations[n : n + num_preds]
                for n in range(0, len(generations), num_preds)
            ]
            generations = grouped_generations

        outputs = dict(
            created=datetime.datetime.now(),
            prompts=inference_state.current_state.get("prompts"),
            generations=generations,
            input_tokens=inference_state.current_state.get("input_tokens"),
        )

        return outputs


class ProcessStreamingOperator(ProcessOutputs):
    output_schema = StreamingOutput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def can_operate(self, inp):
        if inp.get("streaming"):
            return True

    def run(self, inference_state: InferenceState, **kwargs):
        generated_token = inference_state.current_state.get("generated_tokens")[-1]
        generated_logit = inference_state.current_state.get("generated_logits")[-1]
        finish_reason = inference_state.current_state.get("finished_reason")[-1]

        yield_output = super().run(
            generated_tokens=[generated_token],
            generated_logits=[generated_logit],
            finished_reason=[finish_reason],
            inference_state=inference_state,
        )
        yield_output = super().output_schema(**yield_output)
        return {
            "data_to_yield": yield_output,
            "data_to_return": {
                "tokens": kwargs.get("tokens"),
                "kv_cache": kwargs.get("kv_cache"),
                "in_generation": kwargs.get("in_generation"),
                "streaming": True,
            },
        }
