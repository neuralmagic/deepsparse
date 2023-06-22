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

import numpy as np

import pytest
from deepsparse.transformers.utils.kv_cache_ort import KVCacheORT


@pytest.mark.parametrize(
    "state, count, num_tokens, frozen_position, output_state",
    [
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            1,
            3,
            None,
            {"dummy_cache_name": np.array([[[[0], [1], [2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            2,
            3,
            None,
            {"dummy_cache_name": np.array([[[[1], [2], [3]]]])},
        ),
        (
            {"dummy_cache_name": np.array([[[[1], [2], [3]]]])},
            1,
            3,
            0,
            {"dummy_cache_name": np.array([[[[1], [3]]]])},
        ),
    ],
)
def test_kv_cache_ort_shift(state, count, num_tokens, frozen_position, output_state):
    kv_cache = KVCacheORT(
        state=state,
        num_tokens=num_tokens,
        frozen_position=frozen_position,
    )
    kv_cache.shift_last(count=count)
    state = kv_cache.state
    for key in state.keys():
        assert np.array_equal(state[key], output_state[key])


@pytest.mark.parametrize(
    "state, num_tokens",
    [
        (
            {"dummy_cache_name": np.array([[[[0], [0], [1], [2], [3]]]])},
            3,
        ),
    ],
)
def test_kv_cache_ort_reset(
    state,
    num_tokens,
):
    kv_cache = KVCacheORT(
        state=state,
        num_tokens=num_tokens,
    )
    kv_cache.reset()
    for key in kv_cache.state.keys():
        assert np.array_equal(kv_cache.state[key], np.zeros_like(kv_cache.state[key]))
