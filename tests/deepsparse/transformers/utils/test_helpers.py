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
    validate_session_ids,
)


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
