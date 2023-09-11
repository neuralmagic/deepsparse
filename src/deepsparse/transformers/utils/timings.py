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


from dataclasses import dataclass


__all__ = ["TextGenerationTimings"]


@dataclass(frozen=True)
class TextGenerationTimings:
    PROMPT_PREFILL: str = "engine_prompt_prefill"
    PROMPT_PREFILL_SINGLE: str = "engine_prompt_prefill_single"
    TOKEN_GENERATION: str = "engine_token_generation"
    TOKEN_GENERATION_SINGLE: str = "engine_token_generation_single"
    KV_CACHE_UPDATE: str = "kv_cache_update"
