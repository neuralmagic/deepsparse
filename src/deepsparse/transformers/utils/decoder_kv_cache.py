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

from typing import Any, Dict

import numpy


__all__ = ["DecoderKVCache"]


# TODO: Dummy class just to enable testing, will be removed
class KVCache:
    def __init__(self):
        pass


class DecoderKVCache:
    def __init__(self, use_deepsparse_cache: bool = False):
        """
        The goal this object is to handle the manipulation
        of the key value cache.

        :param use_deepsparse_cache: If set to True, the KVCache object
            from the deepsparselib will be loaded as an attribute.
            This object is used to handle the manipulation of the key
            value buffers on the DeepSparse engine side.
        """
        # assuming that kv cache arrays are of shape
        # [batch_size, num_heads, sequence_length, hidden_size]
        self._sequence_axis = 2
        self._session_id = None
        self._freeze_first_position = None
        self._state = None
        self._total_cache_capacity = None
        self._kv_cache = KVCache() if use_deepsparse_cache else None

    def setup_session(
        self,
        session_id: str,
        state: Dict[str, Any],
        freeze_first_position: bool = False,
    ):
        """
        Setup the session - a level of abstraction that allocates
        the resources to store and manipulate the kv cache.

        :param session_id: The session id to use for the current
            session. Used to identify the kv cache state
        :param state: The state of the cache. This is a dictionary
            that maps the name of the cache array to the cache array.
            The cache tensor is a numpy array of shape
            [batch_size, num_heads, sequence_length, hidden_size]
        :param freeze_first_position: If set to True, once the kv cache
            gets filled, the position along the sequence length axis
            that corresponds to the first token will be frozen.
            This assures that, once the KV cache is full (there are no
            "blank" entries), and we are removing the "oldest" entry
            from the cache, we will nevertheless keep the cache entry
            that corresponds to the BOS token in the sequence.
            By default, is set to False.
        """
        self.session_id = session_id
        self._state = state
        self._freeze_first_position = freeze_first_position
        self._total_cache_capacity = state[list(state.keys())[0]].shape[
            self._sequence_axis
        ]

        if self._kv_cache is not None:
            self._kv_cache.reset(
                num_tokens=self._total_cache_capacity,
                freeze_first_position=[int(self._freeze_first_position)],
            )

    def update_session(
        self,
        state: Dict[str, Any],
        num_tokens: int,
        input_ids_len: int,
    ):
        """
        Updating the session is identical with taking the kv cache
        output of from the forward pass and restructuring it, so it
        can be directly used as input for the next forward pass.

        :param state: The state of the cache. This is a dictionary
            that maps the name of the cache array to the cache array.
            The cache tensor is a numpy array of shape
            [batch_size, num_heads, sequence_length, hidden_size]
        :param num_tokens: The number of tokens processed so far,
            corresponding to the number of "non-blank" entries in the
            kv cache array. Even though all numpy arrays in the state
            of the cache have constant sequence length, this length
            corresponds to the total "capacity" of the
            cache. A portion of this memory can be "blank".
        :param input_ids_len: The number of input ids in the current
            input batch: (batch_size, length).
            Corresponds to `input_ids.shape[1]`
        """

        # total_capacity = num_tokens (num of non-blank tokens) +
        # + num_padded_entries (num of blank tokens)
        num_padded_entries = max(0, self._total_cache_capacity - num_tokens)
        # we want to remove input_ids_len entries from the cache
        # because len_input_ids + inp_cache_len = out_cache_len
        num_entries_to_delete = input_ids_len

        while num_entries_to_delete > 0:
            if num_padded_entries > 0:
                """
                Transforms input KV cache that contains blank entries
                Example:
                (entries in the cache denote the order in which they were
                added to the cache, zero is to denote a blank entry)
                ```
                state["state_name"]: (1, 1, 5, 1) = array([[[[0], [0], [1], [2], [3]]]])
                -> num_padded_entries = 2
                -> num_tokens = 3
                -> num_tokens = 3(self._sequence_length = 5)
                -> index to delete -> 1
                self._delete_entry(state, index_to_delete)
                state["state_name"]: (1, 1, 4, 1) = array([[[[0], [1], [2], [3], [0]]]])
                ```
                """
                state = self._delete_entry(state, num_padded_entries - 1)
                num_padded_entries -= 1
            else:
                """
                Transforms the input KV cache that has been totally
                filled with non-blank entries.
                Example:
                ```
                state["state_name"]: (1, 1, 5, 1) = array([[[[1], [2], [3], [4], [5]]]])
                if self.freeze_first_position == False:
                    self._delete_entry(state, 0)
                    state["state_name"]: (1, 1, 4, 1) = array([[[[2], [3], [4], [5]]]])
                else:
                    self._delete_entry(state, 1)
                    state["state_name"]: (1, 1, 4, 1) = array([[[[1], [3], [4], [5]]]])
                ```
                """
                state = self._delete_entry(state, int(self._freeze_first_position))

            num_entries_to_delete -= 1

        self._state = state

    def _delete_entry(self, state: Dict[str, Any], index: int) -> Dict[str, Any]:
        for key, value in state.items():
            state[key] = numpy.delete(value, index, axis=self._sequence_axis)
        return state

    @property
    def session_id(self):
        if self._session_id is None:
            raise ValueError("Attempted to access session_id before setting up session")
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str):
        self._session_id = session_id

    @property
    def cached_inputs(self):
        return self._state
