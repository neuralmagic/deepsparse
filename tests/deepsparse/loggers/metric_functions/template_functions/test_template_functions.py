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
from deepsparse import Pipeline
from tests.utils import mock_engine


yaml_config = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    predefined:
    - func: image_classification
      frequency: 1
    pipeline_inputs.images:
    - func: image_shape
      frequency: 2"""


@pytest.mark.parametrize(
    "yaml_config",
    [
        yaml_config,
    ],
)
@mock_engine(rng_seed=0)
def test_no_function_duplicates_within_template(mock_engine, yaml_config):
    with pytest.raises(ValueError):
        Pipeline.create("image_classification", logger=yaml_config)
