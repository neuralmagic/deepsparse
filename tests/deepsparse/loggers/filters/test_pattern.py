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

import pytest
from deepsparse.loggers.filters import is_match_found, unravel_value_as_generator


class MockValue:
    def __init__(self, data, **kwargs):
        self.data = data
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.mark.parametrize(
    "pattern, string, truth",
    [
        ("re:.*", "foo", True),  # matches everything
        ("re:(?i)operator", "foo", False),
        ("re:(?i)operator", "AddOneOperator", True),
        ("operator", "AddOneOperator", False),
        ("Operator", "AddOneOperator", False),
        ("AddOneOperator", "AddOneOperator", True),
    ],
)
def test_is_match_found(pattern, string, truth):
    assert truth == is_match_found(pattern, string)


@pytest.mark.parametrize(
    "data, expected_output",
    [
        (
            {"a": {"b": 1, "c": {"d": 2}}, "e": 3},
            [("['a']['b']", 1), ("['a']['c']['d']", 2), ("['e']", 3)],
        ),
        (
            [1, [2, [3, 4]], 5],
            [("[0]", 1), ("[1][0]", 2), ("[1][1][0]", 3), ("[1][1][1]", 4), ("[2]", 5)],
        ),
        (
            {"a": 1, "b": "hello", "c": True, "d": 3.14},
            [("['a']", 1), ("['b']", "hello"), ("['c']", True), ("['d']", 3.14)],
        ),
        (
            MockValue(
                {"a": 1, "b": "hello", "c": True, "d": 3.14}, foo=42, bar="example"
            ),
            [
                (".data['a']", 1),
                (".data['b']", "hello"),
                (".data['c']", True),
                (".data['d']", 3.14),
                (".foo", 42),
                (".bar", "example"),
            ],
        ),
    ],
)
def test_unravel_value(data, expected_output):

    assert list(unravel_value_as_generator(data)) == expected_output
