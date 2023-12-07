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

from typing import Any

from pydantic import BaseModel, Field

from deepsparse.transformers.utils import DecoderKVCache
from deepsparse.transformers.utils.helpers import (
    initialize_kv_cache_state,
    prepends_bos_token,
)
from deepsparse.v2.operators import Operator


__all__ = ["KVCacheCreator", "KVCacheCreatorInput"]


class KVCacheCreatorOutput(BaseModel):
    kv_cache: Any = Field(description="KV Cache Created")  # DecoderKVCache


class KVCacheCreatorInput(BaseModel):
    cache_shape: Any = Field(description="shape")
    kv_cache_data_type: Any = Field(description="data type")
    output_names: Any = Field(description="output names")


class KVCacheCreator(Operator):
    input_schema = KVCacheCreatorInput
    output_schema = KVCacheCreatorOutput

    def __init__(
        self,
        tokenizer,
        sequence_length: int,
        prompt_sequence_length: int,
        internal_kv_cache: bool,
    ):
        self.tokenizer = tokenizer
        self.prompt_sequence_length = prompt_sequence_length
        self.internal_kv_cache = internal_kv_cache
        self.sequence_length = sequence_length

    def run(self, cache_shape, kv_cache_data_type: str, output_names: list, **kwargs):
        kv_cache_state = initialize_kv_cache_state(
            cache_shape=cache_shape,
            kv_cache_data_type=kv_cache_data_type,
            output_names=output_names,
            length=self.sequence_length - self.prompt_sequence_length,
            empty=bool(self.internal_kv_cache),
        )

        kv_cache = DecoderKVCache(self.internal_kv_cache)
        kv_cache.setup(
            state=kv_cache_state,
            freeze_first_position=prepends_bos_token(self.tokenizer),
        )
        return {"kv_cache": kv_cache}
