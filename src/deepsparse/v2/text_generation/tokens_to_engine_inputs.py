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

from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["TokensToEngineInputs"]


class TokensToEngineInput(BaseModel):
    tokens: dict = Field(description="tokens")


class TokensToEngineInputs(Operator):
    input_schema = TokensToEngineInput

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        tokens = inp.tokens
        onnx_input_names = pipeline_state.current_state.get("onnx_input_names_no_cache")
        if not all(name in tokens for name in onnx_input_names):
            raise ValueError(
                f"pipeline expected arrays with names {onnx_input_names}, "
                f"received inputs: {list(tokens.keys())}"
            )

        return {"engine_inputs": [tokens[name] for name in onnx_input_names]}, {}
