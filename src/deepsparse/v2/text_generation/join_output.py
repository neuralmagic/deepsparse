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
import numpy

__all__ = ["JoinOutput"]

class JoinOutput(Operator):
    def run(self, inp: Any, context: Optional[Context], inference_state: InferenceState, pipeline_state: PipelineState):
        batch_outputs = [x for x in inp[0]]
        generated_tokens = [x.generated_tokens for x in batch_outputs]
        generated_logits = [x.generated_logits for x in batch_outputs]
        finished_reason = [x.finished_reason for x in batch_outputs]
        generated_tokens = numpy.stack(generated_tokens).squeeze(1)
        generated_logits = numpy.stack(generated_logits).squeeze(1)

      
        return {"generated_tokens": generated_tokens, "generated_logits": generated_logits, "finished_reason": finished_reason}, {}
