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
from deepsparse.loggers.helpers import match


@pytest.mark.parametrize(
    "template, identifier, expected_output",
    [
        ("engine_inputs", "image_classification.engine_inputs", (True, None)),
        (
            "image_classification.engine_inputs",
            "image_classification.engine_inputs",
            (True, None),
        ),
        (
            "pipeline_inputs.images",
            "image_classification.pipeline_inputs",
            (True, "images"),
        ),
        (
            "image_classification.pipeline_inputs.images.something",
            "image_classification.pipeline_inputs",
            (True, "images.something"),
        ),
    ],
)
def test_match(template, identifier, expected_output):
    assert match(template, identifier) == expected_output
