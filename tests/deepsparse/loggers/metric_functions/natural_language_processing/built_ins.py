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
from deepsparse.loggers.metric_functions.utils import BatchResult
from deepsparse.loggers.metric_functions.natural_language_processing import (
    string_length,
)


@pytest.mark.parametrize(
    "string, expected_len",
    [
        ("His palms are sweaty", 20),
        (["knees weak", "arms are heavy"], BatchResult([10, 14])),
        (
            [["knees weak", "arms are heavy"], ["His palms", "are sweaty"]],
            BatchResult([BatchResult([10, 14]), BatchResult([9, 10])]),
        ),
    ],
)
def test_string_length(string, expected_len):
    assert string_length(string) == expected_len
