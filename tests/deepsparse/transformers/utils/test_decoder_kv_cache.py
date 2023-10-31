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
import copy

import numpy as np

import pytest
from deepsparse.transformers.utils import DecoderKVCache


@pytest.mark.parametrize(
    "state, input_ids_len, state_updated, can_continue_updates",
    [
        (  # dummy state of the kv cache
            # needs to be (batch_size, num_heads, seq_len, hidden_size)
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            # number of tokens by which the kv cache is updated
            1,
            # expected updated state
            {"dummy_cache_name": np.array([[[[0], [1], [2], [3]]]])},
            # whether the kv cache is full as a result of the update
            True,
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            1,
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            False,
        ),
        (
            {"dummy_cache_name": np.array([[[[0], [0], [0], [1], [2], [3]]]])},
            3,
            {"dummy_cache_name": np.array([[[[1], [2], [3]]]])},
            True,
        ),
        (
            {"dummy_cache_name": np.array([[[[0], [0], [0], [1], [2], [3]]]])},
            4,
            {"dummy_cache_name": np.array([[[[0], [0], [0], [1], [2], [3]]]])},
            False,
        ),
    ],
)
class TestDecoderKVCache:
    @pytest.fixture
    def setup(
        self,
        state,
        input_ids_len,
        state_updated,
        can_continue_updates,
    ):
        session = DecoderKVCache()

        # compute the number of processed tokens
        state_flattened = state["dummy_cache_name"].flatten()
        num_processed_tokens = state_flattened[state_flattened != 0].shape[0]

        # initialize a session
        session.setup(
            state=state,
            num_processed_tokens=num_processed_tokens,
        )
        yield session, input_ids_len, state_updated, can_continue_updates

    def test_session_attributes(self, setup):
        session, *_ = setup

        # check if the session attributes are set correctly
        state = session.cached_inputs
        assert session.total_num_processed_tokens == np.count_nonzero(
            state["dummy_cache_name"].flatten()
        )
        assert session.capacity == state["dummy_cache_name"].shape[2]

    def test_update_session(self, setup):
        (
            session,
            input_ids_len,
            expected_updated_state,
            expected_can_continue_updates,
        ) = setup
        state = copy.deepcopy(session.cached_inputs)
        # update the session
        can_continue_updates = session.update(state, input_ids_len)
        state_updated = session.cached_inputs
        assert can_continue_updates == expected_can_continue_updates
        for key in state_updated.keys():
            assert np.array_equal(state_updated[key], expected_updated_state[key])

    def test_decrease_capacity_with_overflow(self, setup):
        session_, *_ = setup
        session = copy.deepcopy(session_)
        # setting target_capacity to the number of processed tokens - 1
        # guarantees that the cache buffer will be full
        # after the update
        target_capacity = session.total_num_processed_tokens - 1
        can_continue_updates = session.set_capacity(target_capacity)
        kv_cache_state = session.cached_inputs
        target_array = session_.cached_inputs["dummy_cache_name"]
        assert np.array_equal(target_array, kv_cache_state["dummy_cache_name"])
        assert not can_continue_updates

    def test_decrease_capacity_without_overflow(self, setup):
        session_, *_, can_continue_updates = setup
        if not can_continue_updates:
            pytest.skip(
                "The cache buffer in the setup is already full."
                "It is not possible to decrease the capacity without overflow."
            )
        session = copy.deepcopy(session_)
        capacity = session.capacity
        # setting target_capacity to the number of processed tokens + 1
        # guarantees that the cache buffer will not be full
        # after the update
        target_capacity = session.total_num_processed_tokens + 1
        delta_capacity = capacity - target_capacity
        can_continue_updates = session.set_capacity(target_capacity)
        kv_cache_state = session.cached_inputs
        target_array = session_.cached_inputs["dummy_cache_name"][
            :, :, delta_capacity:, :
        ]
        assert np.array_equal(target_array, kv_cache_state["dummy_cache_name"])
        assert can_continue_updates

    def test_increase_capacity(self, setup):
        session_, *_ = setup
        session = copy.deepcopy(session_)
        capacity = session.capacity
        # increase capacity by 3
        can_continue_updates = session.set_capacity(capacity + 3)
        kv_cache_state = session.cached_inputs
        # check if the capacity has been increased by 3
        assert np.array_equal(
            np.concatenate(
                [[[[[0], [0], [0]]]], session_.cached_inputs["dummy_cache_name"]],
                axis=2,
            ),
            kv_cache_state["dummy_cache_name"],
        )
        # increasing capacity always enables updates
        assert can_continue_updates

    def test_constant_capacity(self, setup):
        session_, *_ = setup
        session = copy.deepcopy(session_)
        capacity = session.capacity
        can_continue_updates = session.set_capacity(capacity)
        kv_cache_state = session.cached_inputs
        assert np.array_equal(
            session_.cached_inputs["dummy_cache_name"],
            kv_cache_state["dummy_cache_name"],
        )
        # keeping constant capacity should enable updates
        # unless the cache buffer is already full
        state_flattened = kv_cache_state["dummy_cache_name"].flatten()
        num_processed_tokens = state_flattened[state_flattened != 0].shape[0]
        assert can_continue_updates == (num_processed_tokens <= capacity)
