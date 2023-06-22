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
from deepsparse.pipeline import ORT_ENGINE
from deepsparse.transformers.utils import DecoderKVCache


@pytest.mark.parametrize(
    "state, input_ids_len, freeze_first_position, state_updated, state_updated_prefill",
    [
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            1,
            False,
            {"dummy_cache_name": np.array([[[[0], [1], [2], [3]]]])},
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[0], [0], [0], [1], [2], [3]]]])},
            2,
            False,
            {"dummy_cache_name": np.array([[[[0], [1], [2], [3]]]])},
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            1,
            False,
            {"dummy_cache_name": np.array([[[[2], [3], [4]]]])},
            {"dummy_cache_name": np.array([[[[1], [2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4]]]])},
            1,
            True,
            {"dummy_cache_name": np.array([[[[1], [3], [4]]]])},
            {"dummy_cache_name": np.array([[[[1], [2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3], [4], [5]]]])},
            2,
            True,
            {"dummy_cache_name": np.array([[[[1], [4], [5]]]])},
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
        state_updated_prefill,
    ):
        decoder = DecoderKVCache(engine_type=ORT_ENGINE)
        state_flattened = state["dummy_cache_name"].flatten()
        num_tokens = state_flattened[state_flattened != 0].shape[0]
        decoder.setup_session(
            session_id="None",
            state=state,
            num_tokens=num_tokens,
            freeze_first_position=freeze_first_position,
        )
        yield decoder, state, num_tokens, input_ids_len, state_updated, state_updated_prefill  # noqa: E501

    def test_with_prefill(self, setup):
        decoder, state, num_tokens, input_ids_len, _, exp_state_updated = setup
        decoder.update_session(
            copy.deepcopy(state), num_tokens, input_ids_len, ignore_generated=True
        )
        state_updated = decoder.cached_inputs
        for key in state_updated.keys():
            assert np.array_equal(state_updated[key], exp_state_updated[key])

    def test_no_prefill(self, setup):
        decoder, state, num_tokens, input_ids_len, exp_state_updated, _ = setup
        decoder.update_session(
            copy.deepcopy(state), num_tokens, input_ids_len, ignore_generated=False
        )
        state_updated = decoder.cached_inputs
        for key in state_updated.keys():
            assert np.array_equal(state_updated[key], exp_state_updated[key])
