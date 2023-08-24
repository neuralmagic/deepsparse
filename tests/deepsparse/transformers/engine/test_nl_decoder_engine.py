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

# TODO: Will have to update those tests now!
from unittest.mock import patch

import numpy as np

from deepsparse.transformers.engines import NLDecoderEngine
from flaky import flaky


class DummyKVCacheDecoder:
    cached_inputs = {
        "past_key_values_1": np.array([10, 11, 12]),
        "past_key_values_2": np.array([13, 14, 15]),
    }


class DummyEngine:
    input_names = ["input_1", "input_2", "past_key_values_1", "past_key_values_2"]


@flaky(max_runs=10, min_passes=1)
def test_generate_token():
    logits = np.array([1.0, 11, 0.9, 0.8])
    expected_token = 1

    with patch.object(NLDecoderEngine, "__init__", lambda x, y, z: None):
        engine = NLDecoderEngine(None, None)
        engine.deterministic = False
        engine.sampling_temperature = 1.0
        token = engine.generate_token(logits)

    assert expected_token == token


def test_add_kv_cache_to_input():
    # keep only the first two inputs
    # (corresponding to "input_1" and "input_2")
    # and add the cached inputs
    # (corresponding to "past_key_values_1" and "past_key_values_2")
    expected_result = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([10, 11, 12]),
        np.array([13, 14, 15]),
    ]

    inp = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
    with patch.object(NLDecoderEngine, "__init__", lambda x, y, z: None):
        nl_decoder_engine = NLDecoderEngine(None, None)
        nl_decoder_engine.engine = DummyEngine()
        nl_decoder_engine.kv_cache = DummyKVCacheDecoder()
        result = nl_decoder_engine.add_kv_cache_to_input(inp)

    for (x, y) in zip(result, expected_result):
        assert np.array_equal(x, y)
