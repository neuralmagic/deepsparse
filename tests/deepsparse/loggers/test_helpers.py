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
from deepsparse.loggers.helpers import check_identifier_match


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
