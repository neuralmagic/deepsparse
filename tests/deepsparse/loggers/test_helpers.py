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
import torch
from deepsparse.loggers.helpers import (
    check_identifier_match,
    get_function_and_function_name,
)
from deepsparse.loggers.metric_functions import identity
from tests.test_data.metric_functions import user_defined_identity


@pytest.mark.parametrize(
    "func, expected_function, expected_function_name",
    [
        ("torch.mean", torch.mean, "mean"),
        ("numpy.max", numpy.max, "max"),
        (
            "torch.distributions.categorical",
            torch.distributions.categorical,
            "distributions.categorical",
        ),
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
    assert get_function_and_function_name(func) == (
        expected_function,
        expected_function_name,
    )


@pytest.mark.parametrize(
    "template, identifier, expected_output",
    [
        ("string_1.string_2", "string_1.string_2", (True, None)),
        ("string_1.string_3", "string_1.string_2", (False, None)),
        (
            "string_1.string_2.string_3.string_4",
            "string_1.string_2",
            (True, "string_3.string_4"),
        ),
        ("re:string_*..*.string.*", "string_1.string_2", (True, None)),
        ("re:string_*..*.string.*", "string_3.string_4", (True, None)),
    ],
)
def test_match(template, identifier, expected_output):
    assert check_identifier_match(template, identifier) == expected_output
