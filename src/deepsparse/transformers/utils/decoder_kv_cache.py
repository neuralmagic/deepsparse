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

from typing import Any, Dict, List

import numpy


__all__ = ["DecoderKVCache", "SEQUENCE_LENGTH_AXIS"]


SEQUENCE_LENGTH_AXIS = 2


class DecoderKVCache:
    def __init__(self, use_deepsparse_cache: bool = False):
        """
        The goal this object is to handle the manipulation
        of the key value cache.

        :param use_deepsparse_cache: If set to True, the `kv_cache` object
            from the deepsparse.LIB will be loaded as an attribute.
            This object is used to handle the manipulation of the
            key/value buffers on the DeepSparse engine side.
        """
        # assuming that kv cache arrays are of shape
        # [batch_size, num_heads, sequence_length, hidden_size]
        self._sequence_len_axis = SEQUENCE_LENGTH_AXIS
        self._use_deepsparse_cache = use_deepsparse_cache
        self._session_id = None
        self._freeze_first_position = None
        self._state = None
        self._total_num_processed_tokens = None
        self._kv_cache = None

    def setup_session(
        self,
        session_id: str,
        state: Dict[str, Any],
        num_processed_tokens: int = 0,
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
            [batch_size, num_heads, sequence_length - num_input_ids, hidden_size]
        :param num_processed_tokens: The number of tokens processed so far.
        :param freeze_first_position: If set to True, once the kv cache
            gets filled, the position along the sequence length axis
            that corresponds to the first token will be frozen.
            This assures that, once the KV cache is full (there are no
            "blank" entries), and we are removing the "oldest" entry
            from the cache, we will nevertheless keep the cache entry
            that corresponds to the BOS token in the sequence.
            By default, is set to False.
        """
        self._session_id = session_id
        self._state = state
        self._freeze_first_position = freeze_first_position
        self._total_num_processed_tokens = num_processed_tokens

        if self._use_deepsparse_cache:
            raise NotImplementedError("DeepSparse cache is not supported yet.")

    def update_session(
        self,
        state: Dict[str, Any],
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
        :param input_ids_len: The number of input ids in the current
            input batch: (batch_size, length).
            Corresponds to `input_ids.shape[1]`
        """
        self._total_num_processed_tokens += input_ids_len
        total_cache_capacity = state[list(state.keys())[0]].shape[
            self._sequence_len_axis
        ]
        # total_capacity = num_tokens (num of non-blank tokens) +
        # + num_padded_entries (num of blank tokens)
        num_padded_entries = max(
            0, total_cache_capacity - self._total_num_processed_tokens
        )
        # we want to remove input_ids_len entries from the cache
        # because len_input_ids + inp_cache_len = out_cache_len
        # TODO: Make it more general once
        # multitoken regression is supported
        num_entries_to_delete = 1  # input_ids_len

        if num_padded_entries:
            """
            Transforms input KV cache that contains blank entries.
            It removes the rightmost blank entries from the cache.
            Example 1:
            (entries in the cache denote the order in which they were
            added to the cache, zero is to denote a blank entry)
            ```
            state["state_name"]: (1, 1, 5, 1) = array([[[[0], [0], [1], [2], [3]]]])
            -> num_padded_entries = 2
            -> num_entries_to_delete = 1
            -> num_padded_entries > num_entries_to_delete
            # there are more blank entries than entries to delete
            results in:
            state["state_name"]: (1, 1, 4, 1) = array([[[[0], [1], [2], [3]]]])
            ```
            Example 2:
            ```
            state["state_name"]: (1, 1, 6, 1) = array([[[[0], [0], [0], [1], [2], [3]]]]) # noqa: E501
            -> num_padded_entries = 3
            -> num_entries_to_delete = 5
            -> num_padded_entries < num_entries_to_delete
            # there are less blank entries than entries to delete
            results in:
            state["state_name"]: (1, 1, 3, 1) = array([[[[1], [2], [3]]]])
            ```
            """
            num_padded_entries_to_delete = min(
                num_padded_entries, num_entries_to_delete
            )
            idxs_to_remove = [
                num_padded_entries - i - 1 for i in range(num_padded_entries_to_delete)
            ]
            # if we had fewer blank entries than entries to delete,
            # we updated the number of entries to delete to a non-zero value
            num_entries_to_delete = max(0, num_entries_to_delete - num_padded_entries)
            # update the state of the cache
            state = self._delete_entries(state, idxs_to_remove)

        if num_entries_to_delete:
            """
            Transforms the input KV cache that has been totally
            filled with non-blank entries.
            Example:
            ```
            state["state_name"]: (1, 1, 5, 1) = array([[[[1], [2], [3], [4], [5]]]])
            num_entries_to_delete = 2
            if self.freeze_first_position == False:
                state["state_name"]: (1, 1, 3, 1) = array([[[[3], [4], [5]]]])
            else:

                state["state_name"]: (1, 1, 3, 1) = array([[[[1], [4], [5]]]])
            ```
            """
            idxs_to_remove = [
                i + int(self._freeze_first_position)
                for i in range(num_entries_to_delete)
            ]

            state = self._delete_entries(state, idxs_to_remove)

        self._state = state

    def _delete_entries(
        self, state: Dict[str, Any], indices: List[int]
    ) -> Dict[str, Any]:
        for key, value in state.items():
            state[key] = numpy.delete(value, indices, axis=self._sequence_len_axis)
            state[key] = numpy.ascontiguousarray(state[key])
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
