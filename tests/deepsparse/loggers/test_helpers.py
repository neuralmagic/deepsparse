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

from typing import Any

import numpy
from pydantic import BaseModel

import pytest
from deepsparse.loggers.helpers import (
    access_nested_value,
    check_identifier_match,
    get_function_and_function_name,
    possibly_extract_value,
)
from deepsparse.loggers.metric_functions import identity
from tests.test_data.metric_functions import user_defined_identity


@pytest.mark.parametrize(
    "func, expected_function, expected_function_name",
    [
        ("numpy.max", numpy.max, "max"),
        ("numpy.linalg.norm", numpy.linalg.norm, "linalg.norm"),
        (
            "tests/test_data/metric_functions.py:user_defined_identity",
            user_defined_identity,
            "user_defined_identity",
        ),
        ("identity", identity, "identity"),
    ],
)
def test_get_function_and_function_name(
    func, expected_function, expected_function_name
):
    assert get_function_and_function_name(func)[1] == expected_function_name


@pytest.mark.parametrize(
    "template, identifier,expected_output",
    [
        ("string_1/string_2[1:3]", "string_1/string_2", (True, "[1:3]")),
        ("string_1/string_2", "string_1/string_2", (True, None)),
        ("string_1/string_3", "string_1/string_2", (False, None)),
        (
            "string_1/string_2/string_3.string_4",
            "string_1/string_2",
            (True, "string_3.string_4"),
        ),
        ("re:string_*..*.string.*", "string_1/string_2", (True, None)),
        ("re:string_*..*.string.*", "string_3/string_4", (True, None)),
    ],
)
def test_check_identifier_match(template, identifier, expected_output):
    assert check_identifier_match(template, identifier) == expected_output


class MockModel__(BaseModel):
    key_3: Any


class MockModel_(BaseModel):
    key_2: Any


class MockModel(BaseModel):
    key_1: Any


value_1 = MockModel(key_1=MockModel_(key_2=[0, 1, 2, 3]))
value_2 = MockModel(key_1=[[MockModel_(key_2=[0, MockModel__(key_3=5)])]])
value_3 = MockModel(
    key_1=[[MockModel_(key_2={"key_3": [{"key_4": [0, 1, 2, 3, 4, 5]}, 0]})]]
)


@pytest.mark.parametrize(
    "value, remainder, expected_value",
    [
        (value_1, "key_1.key_2[2]", 2),
        (value_2, "key_1[0][0].key_2[1].key_3", 5),
        (value_2, "key_1[0][0].key_2[0:2][1].key_3", 5),
        (value_3, "key_1[0][0].key_2['key_3'][0]['key_4'][2:4]", [2, 3]),
    ],
)
def test_possibly_extract_value(value, remainder, expected_value):
    assert expected_value == possibly_extract_value(value=value, remainder=remainder)


@pytest.mark.parametrize(
    "value, square_brackets, expected_value",
    [
        ([0, 1, 2, 3], "2", 2),
        ([0, 1, 2, 3], "0:3", [0, 1, 2]),
        ([[0, 1, 2, 3]], "0, 0:3", [0, 1, 2]),
        ([[0, 1, 2, 3]], "0, -1", 3),
        ([[0, 1, 2, 3]], "0, 0:3, 2", 2),
        ([[0, 1, 2, 3]], "0, 0:3, 2", 2),
        ([0, 1, 2, 3, 4], "1:-2, 0", 1),
        ([[0, 1, 2, 3, 4]], "0, 1:-1:1", [1, 2, 3]),
        (numpy.array([[0, 1, 2, 3, 4]]), "0, 1:-1:1", numpy.array([1, 2, 3])),
        ({"key_1": {"key_2": [3]}}, "'key_1','key_2',0", 3),
        ({"key_1": [0, 1, {"key_2": [3, 4, 5]}]}, "'key_1',2,'key_2',0", 3),
    ],
)
def test_access_nested_value(value, square_brackets, expected_value):
    if isinstance(expected_value, numpy.ndarray):
        assert numpy.array_equal(
            expected_value, access_nested_value(value, square_brackets)
        )
        return

    assert expected_value == access_nested_value(
        value=value, square_brackets=square_brackets
    )
