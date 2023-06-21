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

from typing import Dict, Optional

import numpy


__all__ = ["KVCacheORT"]


class KVCacheORT:
    """
    Interface for the onnxruntime KV Cache object. Its goal
    is to mimic the behavior of the KVCacheDeepSparse object.
    The interface consumes a KV cache dictionary
    returned by the ORT engine and provide a way to:
     -  shift the cache, so the resulting state can be used
        autoregressively as an input to the next generation step
    -   reset the cache

    :param state: The state of the cache. This is a dictionary
        that maps the name of the cache array to the cache array.
        The cache tensor is a numpy array of shape
        [batch_size, num_heads, sequence_length, hidden_size]
    :param num_tokens: The number of tokens processed so far,
        corresponding to the number of "non-blank" entries in the
        kv cache array. Even though all numpy arrays in the state
        of the cache have constant sequence length, this length
        corresponds to the total "capacity" or "memory" of the
        cache. A portion of this memory can be "blank".

        Examples
        (entries in the cache denote the order in which they were
        added to the cache, zero is to denote a blank entry)
        ```
        state["state_name"]: (1, 1, 5, 1) = array([[[[0],[0],[1],[2],[3]]]])
        -> num_tokens = 3 (self._sequence_length = 5)
        ```
        ```
        state["state_name"]: (1, 1, 3, 1) = array([[[[1],[2],[3]]]])
        -> num_tokens = 3 (self._sequence_length = 3)
        ```

    :param frozen_position: The position along the sequence length axis
        that is frozen and thus, once it is occupied by a "non-blank"
        cache entry, it cannot be removed from the cache.
        The goal is, once the KV cache is full (there no "blank" entries),
        to keep some cache positions frozen, as they may
        correspond to tokens that we may not want to "pop" from the
        kv cache e.g. the BOS token in the sequence. By default, is None,
        this means that we do not freeze any cache positions.

    :param sequence_axis: The axis of the sequence length in the cache
        array. By default, this is 2, which corresponds to the sequence
        length axis in the cache array.
    """

    def __init__(
        self,
        state: Dict[str, numpy.ndarray],
        num_tokens: int,
        frozen_position: Optional[int] = None,
        sequence_axis: int = 2,
    ):
        self._state = state
        self._num_tokens = num_tokens
        self._frozen_position = frozen_position
        self._sequence_axis = sequence_axis
        self._sequence_length = state[list(state.keys())[0]].shape[sequence_axis]

    @property
    def state(self) -> Dict[str, numpy.ndarray]:
        """
        Property for the state of the cache

        :return: The state of the cache
        """
        return self._state

    @state.setter
    def state(self, value: Dict[str, numpy.ndarray]):
        """
        Setter for the state of the cache
        :param value: The new state of the cache
        """
        self._state = value

    def shift_last(self, count: int = 1):
        """
        Shift the last #count entries in the cache,
        so the resulting state can be used autoregressively
        as an input to the next generation step.

        Examples:
        ```
        self.state["state_name"] = array([[[[0],[0],[1],[2],[3]]]])
        -> self.shift_last(1)
        self.state["state_name"] = array([[[[0],[1],[2],[3]]]])
        ```

        ```
        self.state["state_name"] = array([[[[0],[0],[1],[2],[3]]]])
        -> self.shift_last(2)
        self.state["state_name"] = array([[[[1],[2],[3]]]])
        ```

        ```
        self._frozen_position = 0
        self.state["state_name"] = array([[[[1],[2],[3]]]])
        -> self.shift_last(1)
        self.state["state_name"] = array([[[[1],[3]]]])
        ```

        :param count: The number of generated tokens
            (new "non-blank" entries added to the cache),
            that have been generated since the construction
            of the KVCacheORT object. By default, this is 1
            (we shift by one token in the cache).
        """
        for c in range(count):
            index_to_remove = 0

            # the number of "non-blank" entries in the cache is
            # the sum of the "non-blank" token present at the
            # construction of the KVCacheORT object and the number
            # of tokens generated since
            num_processed_tokens = self._num_tokens + c

            # if the cache is full, we want to keep the frozen position
            # i.e. delete the index that follows the frozen position
            if num_processed_tokens >= self._sequence_length:
                index_to_remove = (
                    self._frozen_position + 1
                    if self._frozen_position is not None
                    else 0
                )

            for cache_name, cache_value in self.state.items():
                self.state[cache_name] = numpy.delete(
                    cache_value, index_to_remove, self._sequence_axis
                )

    def reset(self):
        """
        Reset the cache to its initial state
        """
        self._state = {
            keys: numpy.zeros_like(value) for keys, value in self._state.items()
        }
