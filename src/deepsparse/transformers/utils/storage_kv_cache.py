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
from typing import Dict, Union

from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache


_LOGGER = logging.getLogger(__name__)

__all__ = ["SessionStorageKVCache"]


class SessionStorageKVCache:
    """
    A storage that stores the kv cache sessions.
    Each session is a DecoderKVCache object that
    stores the state of the kv cache.
    The storage is a dictionary that where keys are session_ids
    and values are of all the active sessions.
    """

    def __init__(self):
        self._memory: Dict[str, DecoderKVCache] = dict()

    def __len__(self):
        return len(self._memory)

    def __str__(self):
        return (
            f"{SessionStorageKVCache.__name__}:\n "
            f"\tsessions: {[session_name for session_name in self._memory.keys()]}\n"
        )

    def has_session(self, session_id: str) -> bool:
        """
        Check if the storage has a session with the given session id.
        :param session_id: The identifier of the cache session.
        :return: True if the storage has a session with the given session id.
        """
        return session_id in self._memory

    def put(self, session: DecoderKVCache):
        """
        Put the cache session in the storage.

        :param session: The session to store.
        """
        session_id = session.id
        if self.has_session(session_id):
            _LOGGER.debug(
                f"Session: {session_id} already exists in the storage. "
                f"It will be overwritten."
            )
        self._memory[session.id] = session

    def get(self, session_id: str) -> Union[DecoderKVCache, None]:
        """
        Get the state of the kv cache for a session from the storage.

        :param session_id: The identifier of the cache session.
        :return: The state of the kv cache for the session.
        """
        session = self._memory.get(session_id)
        if session is None:
            _LOGGER.debug(f"No cache session found for session id: {session_id}")
        return session

    def pop(self, session_id: str):
        """
        Remove the session from the storage.

        :param session_id: The identifier of the cache session.
        """
        popped_element = self._memory.pop(session_id, None)
        if popped_element is None:
            raise ValueError(
                f"Attempting to remove session: {session_id} from the storage. "
                f"However, the session does not exist in the storage."
            )
