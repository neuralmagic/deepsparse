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
    "state, input_ids_len, freeze_first_position, state_updated",
    [
        (  # dummy state of the kv cache
            # needs to be (batch_size, num_heads, seq_len, hidden_size)
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            # number of tokens by which the kv cache is updated
            1,
            # whether the first position of the kv cache is frozen
            False,
            # expected updated state
            {"dummy_cache_name": np.array([[[[0], [1], [2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            1,
            False,
            {"dummy_cache_name": np.array([[[[2], [3], [4]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            1,
            True,
            {"dummy_cache_name": np.array([[[[1], [3], [4]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            3,
            False,
            {"dummy_cache_name": np.array([[[[2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            3,
            False,
            {"dummy_cache_name": np.array([[[[4]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [3], [4], [5]]]])},
            2,
            True,
            {"dummy_cache_name": np.array([[[[1], [5]]]])},
        ),
    ],
)
class TestDecoderKVCache:
    @pytest.fixture
    def setup(
        self,
        state,
        input_ids_len,
        freeze_first_position,
        state_updated,
    ):
        session = DecoderKVCache()

        # compute the number of processed tokens
        state_flattened = state["dummy_cache_name"].flatten()
        num_processed_tokens = state_flattened[state_flattened != 0].shape[0]

        # initialize a session
        session.setup(
            session_id="dummy_id",
            state=state,
            num_processed_tokens=num_processed_tokens,
            freeze_first_position=freeze_first_position,
        )
        yield session, input_ids_len, state_updated

    def test_session_attributes(self, setup):
        session, _, _ = setup

        # check if the session attributes are set correctly
        state = session.cached_inputs
        assert session.capacity == state["dummy_cache_name"].shape[2]
        assert session.id == "dummy_id"

    def test_set_capacity(self, setup):
        session, _, _ = setup

        # check if the capacity is set correctly
        self._test_increase_capacity(session)  # increase
        self._test_decrease_capacity(session)  # decrease
        # self._test_constant_capacity(session)  # constant

    def test_update_session(self, setup):
        session, input_ids_len, expected_updated_state = setup
        state = copy.deepcopy(session.cached_inputs)
        # update the session
        session.update(state, input_ids_len)
        state_updated = session.cached_inputs
        for key in state_updated.keys():
            assert np.array_equal(state_updated[key], expected_updated_state[key])

    @staticmethod
    def _test_increase_capacity(session_):
        session = copy.deepcopy(session_)
        capacity = session.capacity
        # increase capacity by 3
        session.set_capacity(capacity + 3)
        kv_cache_state = session.cached_inputs
        # check if the capacity has been increased by 3
        assert np.array_equal(
            np.concatenate(
                [[[[[0], [0], [0]]]], session_.cached_inputs["dummy_cache_name"]],
                axis=2,
            ),
            kv_cache_state["dummy_cache_name"],
        )

    @staticmethod
    def _test_decrease_capacity(session_):
        session = copy.deepcopy(session_)
        capacity = session.capacity
        # decrease capacity by 3
        session.set_capacity(capacity - 3)
        kv_cache_state = session.cached_inputs
        # check if the capacity has been decreased by 3
        if session_._freeze_first_position:
            bos_token = session_.cached_inputs["dummy_cache_name"][:, :, :1, :]
            new_array = session_.cached_inputs["dummy_cache_name"][:, :, 4:, :]
            target_array = np.concatenate([bos_token, new_array], axis=2)
        else:
            target_array = session_.cached_inputs["dummy_cache_name"][:, :, 3:, :]
        assert np.array_equal(target_array, kv_cache_state["dummy_cache_name"])

    @staticmethod
    def _test_constant_capacity(session_):
        session = copy.deepcopy(session_)
        capacity = session.capacity
        session.set_capacity(capacity)
        kv_cache_state = session.cached_inputs
        assert np.array_equal(
            session_.cached_inputs["dummy_cache_name"],
            kv_cache_state["dummy_cache_name"],
        )
