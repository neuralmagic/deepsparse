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

from typing import Any, Optional

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState

__all__ = ["PrepareforMultiEngine"]

class PrepareforMultiEngine(Operator):
    def __init__(self, prompt_sequence_length):
        self.prompt_sequence_length = prompt_sequence_length

    def can_operate(self, inp: Any, inference_state: InferenceState):
        tokens = inp.tokens
        if len(tokens) > self.prompt_sequence_length:
            print("can process", len(tokens))
            return True
        return False

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        tokens = inp.tokens
        num_batches = len(tokens) // self.prompt_sequence_length
        state_update = {
            "multitoken_num_batches": num_batches,
            "start_multi_token": 0,
            "end_multi_token": self.prompt_sequence_length,
            "batches_processed": 0,
            "num_tokens_processed": 0,
        }
        return inp, state_update
