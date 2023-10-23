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

from deepsparse.transformers.utils import DecoderKVCache
from deepsparse.v2.operators import Operator
from deepsparse.v2.utils import Context, InferenceState, PipelineState

__all__ = ["PrepareforPrefill"]

class PrepareforPrefillOutput(BaseModel):
    tokens: list = Field(description="tokens")
    kv_cache: Any = Field(description="KV Cache") #DecoderKVCache


class PrepareforPrefill(Operator):
    output_schema = PrepareforPrefillOutput

    def __init__(
        self, kv_cache_creator
    ):
        self.kv_cache_creator = kv_cache_creator

    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ):
        cache_shape = pipeline_state.current_state.get("cache_shape")
        data_type = pipeline_state.current_state.get("data_type")
        output_names = pipeline_state.current_state.get("output_names")

        engine_inputs = inp.get("engine_inputs")
        kv_cache, _ = self.kv_cache_creator(
            context=context,
            pipeline_state=pipeline_state,
            inference_state=inference_state,
            **{
                "cache_shape": cache_shape,
                "kv_cache_data_type": data_type,
                "output_names": output_names,
            },
        )
        tokens = engine_inputs[0][engine_inputs[1].nonzero()].tolist() # wont work for multiple sequences
        return {"tokens": tokens, "kv_cache": kv_cache.kv_cache}, {}
