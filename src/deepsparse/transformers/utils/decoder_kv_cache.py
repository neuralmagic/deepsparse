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

from typing import Any, Dict, Optional

from deepsparse.pipeline import SUPPORTED_PIPELINE_ENGINES
from deepsparse.transformers.utils.kv_cache_ort import KVCacheORT


__all__ = ["DecoderKVCache"]


class DecoderKVCache:
    def __init__(self, engine_type: str):
        """
        The goal of DecoderKVCache is to provide a common
        interface for the KVCache objects used
        by the NLDecoderEngine

        :param engine_type: The engine type to use for the decoder
        """
        if engine_type not in SUPPORTED_PIPELINE_ENGINES:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        elif engine_type != "onnxruntime":
            raise NotImplementedError(f"Unsupported engine type: {engine_type}")
        self._kv_cache_type = KVCacheORT

        self._kv_cache = None
        self._session_id = None
        self._frozen_position = None
        self._num_tokens = None

    def setup_session(
        self,
        session_id: str,
        state: Dict[str, Any],
        num_tokens: int,
        frozen_position=Optional[int],
    ):
        """
        Setup the session that will be used to transform
        the input and output cache values

        :param session_id: The session id to use for the current
            session
        :param state: The state of the cache. This is a dictionary
            that maps the name of the cache array to the cache array.
            The cache tensor is a numpy array of shape
            [batch_size, num_heads, sequence_length, hidden_size]
        :param num_tokens: The number of tokens processed so far,
            corresponding to the number of "non-blank" entries in the
            kv cache array.
        :param frozen_position: The position along the sequence length axis
            that is frozen and thus, once it is occupied by a "non-blank"
            cache entry, it cannot be removed from the cache.
        """
        self.session_id = session_id
        self._num_tokens = num_tokens
        self._frozen_position = frozen_position
        self._initialize_kv_cache(state)

    def update_session(self, state: Dict[str, Any]):
        """
        Update the session with the new state of the cache

        :param state: The state of the cache. This is a dictionary
            that maps the name of the cache array to the cache array.
            The cache tensor is a numpy array of shape
            [batch_size, num_heads, sequence_length, hidden_size]
        """
        self._num_tokens += 1
        self._initialize_kv_cache(state)

    @property
    def session_id(self):
        if self._session_id is None:
            raise ValueError("Attempted to access session_id before setting up session")
        return self._session_id

    @property
    def cached_inputs(self):
        if self._kv_cache is None:
            raise ValueError(
                "Attempted to access cached inputs before setting up session"
            )
        # TODO: Not sure whether this is the appropriate place
        # to invoke the shift_last method, to reconsider
        self._kv_cache.shift_last()
        return self._kv_cache.state

    @session_id.setter
    def session_id(self, session_id: str):
        self._session_id = session_id

    def _initialize_kv_cache(self, state: Dict[str, Any]):
        self._kv_cache = KVCacheORT(
            state=state,
            num_tokens=self._num_tokens,
            frozen_position=self._frozen_position,
        )
