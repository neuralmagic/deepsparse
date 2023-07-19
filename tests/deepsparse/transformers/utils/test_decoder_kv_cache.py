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
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            1,
            False,
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
        decoder = DecoderKVCache()
        state_flattened = state["dummy_cache_name"].flatten()
        num_processed_tokens = state_flattened[state_flattened != 0].shape[0]
        decoder.setup_session(
            session_id="None",
            state=state,
            num_processed_tokens=num_processed_tokens,
            freeze_first_position=freeze_first_position,
        )
        yield decoder, state, input_ids_len, state_updated

    def test_update_session(self, setup):
        decoder, state, input_ids_len, exp_state_updated = setup
        decoder.update_session(copy.deepcopy(state), input_ids_len)
        state_updated = decoder.cached_inputs
        for key in state_updated.keys():
            assert np.array_equal(state_updated[key], exp_state_updated[key])
