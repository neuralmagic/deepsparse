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

import os

import yaml

import pytest
from deepsparse.loggers.config import PipelineLoggingConfig
from deepsparse.loggers.metric_functions.helpers import (
    data_logging_config_from_predefined,
)


@pytest.mark.parametrize(
    "group_name, frequency, path_to_config",
    [
        (
            "image_classification",
            1,
            "examples/data-logging-configs/image_classification.yaml",
        ),
        ("object_detection", 2, "examples/data-logging-configs/object_detection.yaml"),
        (
            "question_answering",
            3,
            "examples/data-logging-configs/question_answering.yaml",
        ),
    ],
)
def test_data_config_from_predefined(group_name, frequency, path_to_config):
    # if GENERATE_CONFIGS is set, then generate new, fresh
    # configs to the `path_to_config`
    save_dir, save_name = os.path.dirname(path_to_config), os.path.basename(
        path_to_config
    )
    config = data_logging_config_from_predefined(
        group_names=group_name,
        frequency=frequency,
        save_name=save_name,
        save_dir=save_dir if os.environ.get("NM_GENERATE_CONFIGS") else None,
    )
    with open(path_to_config, "r") as f:
        expected_config = yaml.safe_load(f)
    assert PipelineLoggingConfig(**expected_config) == PipelineLoggingConfig(
        **yaml.safe_load(config)
    )
