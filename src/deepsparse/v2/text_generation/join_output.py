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

from typing import List

import numpy

from deepsparse.transformers.utils.helpers import pad_to_fixed_length
from deepsparse.v2.operators import Operator
from deepsparse.v2.text_generation.compile_generations import CompileGenerationsOutput


__all__ = ["JoinOutput"]


class JoinOutput(Operator):
    """
    Run this operator to combine the results from multiple prompts.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def run(self, inp: List[CompileGenerationsOutput], **kwargs):
        batch_outputs = [x for x in inp[0]]
        generated_tokens = [x.generated_tokens for x in batch_outputs]
        generated_logits = [x.generated_logits for x in batch_outputs]
        finished_reason = [x.finished_reason for x in batch_outputs]

        max_len = max(token.shape[1] for token in generated_tokens)

        # pad all tokens to the same length
        tokens = [
            pad_to_fixed_length(
                array=prediction,
                max_len=max_len,
                value=self.tokenizer.pad_token_id,
                axis=1,
            )
            for prediction in generated_tokens
        ]

        # find the longest sequence in the batch of logits
        max_len = max(logits.shape[1] for logits in generated_logits)

        # pad all logits to the same length
        logits = [
            pad_to_fixed_length(array=single_logits, max_len=max_len, axis=1)
            for single_logits in generated_logits
        ]

        tokens = numpy.concatenate(tokens)
        logits = numpy.concatenate(logits)

        return {
            "generated_tokens": tokens,
            "generated_logits": logits,
            "finished_reason": finished_reason,
        }
