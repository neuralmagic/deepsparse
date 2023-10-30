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

from pydantic import BaseModel, Field

from deepsparse.transformers.utils.token_generator import TokenGenerator
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["TokenGeneratorOperator"]


class TokenGeneratorOperatorInput(BaseModel):
    logits_shape: Any = Field(description="shape")
    tokens: Any = Field(description="tokens", default=[])
    deterministic: bool = Field(description="deterministic", default=False)
    sampling_temperature: int = Field(description="sampling temperature", default=1)
    kwargs: dict = Field(description="kwargs", default={})


class TokenGeneratorOperator(Operator):
    input_schema = TokenGeneratorOperatorInput

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        inference_state: InferenceState,
        pipeline_state: PipelineState,
    ):
        token_generator = TokenGenerator(
            logits_shape=inp.logits_shape,
            deterministic=inp.deterministic,
            sampling_temperature=inp.sampling_temperature,
            **inp.kwargs,
        )

        return {"token_generator": token_generator}, {}
