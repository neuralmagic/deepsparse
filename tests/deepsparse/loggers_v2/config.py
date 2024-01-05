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
from deepsparse.loggers_v2.config import LoggingConfig


def test_config_inputs():
    config_data = {
        "system": {"level": "info"},
        "performance": {"frequency": 0.1, "timings": True, "cpu": True},
        "metrics": {
            "process_input.prompt": {"function": "identity", "frequency": 0.1},
            "prep_for_generation.logits": {"function": "max", "frequency": 1.0},
            "prep_for_generation.bar": {"function": "max", "frequency": 1.0},
        },
    }
    logging_config = LoggingConfig(**config_data)
    assert logging_config.dict() == config_data


def test_op_name_and_key_failure():
    config_data = {
        "system": {"level": "INFO"},
        "performance": {"frequency": 0.1, "timings": True, "cpu": True},
        "metrics": {
            "process_input": {"function": "identity", "frequency": 0.1},
        },
    }

    with pytest.raises(ValueError):
        LoggingConfig(**config_data)
