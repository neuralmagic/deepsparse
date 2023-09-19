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

from typing import Optional
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline, TextGenerationOutput, TextGenerationInput
from deepsparse.transformers.utils import SessionStorageKVCache, DecoderKVCache
from pydantic import Field

class ChatOutput(TextGenerationOutput):
    session_id: Optional[str] = Field(
        default=None, description="A string identifier for the kv cache session."

class ChatInput(TextGenerationInput):
    session_id: Optional[str] = Field(
        default=None, description="A string identifier for the kv cache session."
    )

class ChatPipeline(TextGenerationPipeline):
    def __init__(self, **kwargs):
        self.session_storage = SessionStorageKVCache()
        super().__init__(**kwargs)



    def get_decoder_kv_cache(self, context) -> Optional[DecoderKVCache]:
        session_id = context.get("session_id", None)
        session = self.session_storage.get(session_id)
        if session is None:
            session = self._create_decoder(...)
        return session

    def process_inputs(...):

        engine_input, context = super().process_inputs(...)
        # add session_id context
        return engine_input, context

    def split_engine_inputs(...):
        pass
