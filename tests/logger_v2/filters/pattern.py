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
from deepsparse.loggers_v2.filters import is_match_found


@pytest.mark.parametrize(
    "pattern, string, truth",
    [
        (".*", "foo", True),  # matches everything
        (r"(?i)operator", "foo", False),
        (r"(?i)operator", "AddOneOperator", True),
    ],
)
def test_is_match_found(pattern, string, truth):
    assert truth == is_match_found(pattern, string)
