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

import numpy

import pytest
from deepsparse.transformers.utils.helpers import (
    create_causal_mask,
    initialize_kv_cache_state,
    validate_session_ids,
)


@pytest.mark.parametrize(
    "input_ids, attention_mask, expected_causal_mask",
    [
        (
            numpy.array([[8]]),
            numpy.array([[0, 0, 1, 1, 1]]),
            numpy.array([[0, 0, 1, 1, 1]]),
        ),
        (
            [8],
            [0, 0, 1, 1, 1],
            numpy.array([[0, 0, 1, 1, 1]]),
        ),
        (
            numpy.array([[1, 2, 3, 4]]),
            numpy.array([[1, 1, 1, 1, 1, 1]]),
            numpy.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
        (
            numpy.array([[1, 2, 3, 4]]),
            numpy.array([[0, 0, 0, 1, 1, 1]]),
            numpy.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1],
                ]
            ),
        ),
        (
            numpy.array([[1, 2, 3]]),
            numpy.array(
                [
                    [
                        0,
                        1,
                        1,
                        1,
                    ]
                ]
            ),
            numpy.array(
                [
                    [0, 1, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 1],
                ]
            ),
        ),
        (
            [1, 2, 3],
            [0, 1, 1, 1],
            numpy.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
        ),
    ],
)
def test_create_causal_mask(input_ids, attention_mask, expected_causal_mask):
    causal_mask = create_causal_mask(input_ids, attention_mask)
    assert numpy.array_equal(causal_mask, expected_causal_mask[None, None, ...])


@pytest.mark.parametrize(
    "cache_shape, kv_cache_data_type, output_names, length, empty, expected_result",
    [
        (
            (1, 2, 3, 4),
            numpy.float32,
            ["present.1", "present.2", "present.3"],
            None,
            False,
            {
                "past_key_values.1": numpy.zeros((1, 2, 3, 4)),
                "past_key_values.2": numpy.zeros((1, 2, 3, 4)),
                "past_key_values.3": numpy.zeros((1, 2, 3, 4)),
            },
        ),
        (
            (5, 6, 7, 8),
            numpy.int8,
            ["present.1", "present.2"],
            10,
            True,
            {
                "past_key_values.1": numpy.zeros((5, 6, 10, 8), dtype=numpy.int8),
                "past_key_values.2": numpy.zeros((5, 6, 10, 8), dtype=numpy.int8),
            },
        ),
    ],
)
def test_initialize_kv_cache_state(
    cache_shape, kv_cache_data_type, output_names, length, empty, expected_result
):
    # make sure that resulting Dict[str, numpy.ndarray] is the same
    # as the expected_result
    result = initialize_kv_cache_state(
        cache_shape, kv_cache_data_type, output_names, length, empty
    )
    assert result.keys() == expected_result.keys()
    for key in result.keys():
        assert numpy.array_equal(result[key], expected_result[key])


@pytest.mark.parametrize(
    "sequences, session_ids, result, should_raise_error",
    [
        ("sequence", None, None, False),
        (["sequence"], None, None, False),
        (["sequence_1", "sequence_2"], None, None, False),
        ("sequence", "session_id", ["session_id"], False),
        (["sequence"], "session_id", ["session_id"], False),
        (["sequence"], ["session_id"], ["session_id"], False),
        (["sequence_1", "sequence_2"], "session_id", None, True),
        (
            ["sequence_1", "sequence_2"],
            ["session_id_1", "session_id_2", "session_id_3"],
            None,
            True,
        ),
        (["sequence_1", "sequence_2"], ["session_id_1", "session_id_1"], None, True),
    ],
)
def test_validate_session_ids(sequences, session_ids, result, should_raise_error):
    if should_raise_error:
        with pytest.raises(ValueError):
            validate_session_ids(session_ids, dict(sequences=sequences))
    else:
        assert result == validate_session_ids(session_ids, dict(sequences=sequences))
