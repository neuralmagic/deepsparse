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

import logging
import time
from typing import Set, Union

from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache


_LOGGER = logging.getLogger(__name__)

__all__ = ["SessionStorageKVCache"]


class SessionStorageKVCache:
    """
    A storage that stores the kv cache sessions.
    Each session is a DecoderKVCache object that stores the state of the kv cache.
    The storage is a set of all the active sessions.
    """

    def __init__(self):
        self._memory: Set[DecoderKVCache] = set()
        self.timestamp = time.time()

    def __len__(self):
        return len(self._memory)

    def __str__(self):
        return (
            f"{SessionStorageKVCache.__name__}:\n "
            f"\tsessions: {[session.id for session in self._memory]}\n"
        )

    def update_timestamp(func):
        def wrapper(self, *args, **kwargs):
            self.timestamp = time.time()
            return func(self, *args, **kwargs)

        return wrapper

    def has_session(self, session_id: str) -> bool:
        """
        Check if the storage has a session with the given session id.
        :param session_id: The identifier of the cache session.
        :return: True if the storage has a session with the given session id.
        """
        return any(session.id == session_id for session in self._memory)

    @property
    def internal_cache_active(self) -> bool:
        """
        Check if the internal cache is active for sessions in the storage.
        Note: This assumes that all the sessions in
        the storage have the same internal cache state.

        :return: True if the internal cache is active
            for any of the sessions in the storage.
        """
        if len(self._memory) == 0:
            raise ValueError(
                "Attempting to determine if internal cache is active for "
                "sessions of the KV cache storage. However, the storage is empty."
            )
        return next(iter(self._memory)).engine_internal_cache is not None

    @update_timestamp
    def put(self, session: DecoderKVCache):
        """
        Put the cache session in the storage.

        :param session: The session to store.
        """
        session_id = session.id
        old_session = self._get(session_id)
        if old_session is not None:
            self._memory.remove(old_session)
        self._memory.add(session)

    def get(self, session_id: str) -> Union[DecoderKVCache, None]:
        """
        Get the state of the kv cache for a session from the storage.

        :param session_id: The identifier of the cache session.
        :return: The state of the kv cache for the session.
        """
        return self._get(session_id)

    def _get(self, session_id: str) -> Union[DecoderKVCache, None]:
        session = next(
            (session for session in self._memory if session.id == session_id),
            None,
        )
        if session is None:
            _LOGGER.debug(f"No cache session found for session id: {session_id}")
        return session
