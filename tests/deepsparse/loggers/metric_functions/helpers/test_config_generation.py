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

import pytest
from deepsparse.loggers.config import PipelineLoggingConfig
from deepsparse.loggers.metric_functions.helpers.config_generation import (
    data_logging_config_from_predefined,
)


config_1 = """
loggers:
    python:
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 1
    ..."""

config_2 = """
loggers:
    python:
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 1
    ..."""

config_3 = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 2
    ..."""


@pytest.mark.parametrize(
    "group_names, loggers, frequency, save_dir, expected_config",
    [
        ("image_classification", None, 1, None, config_1),
        (["image_classification"], None, 1, None, config_1),
        (["image_classification", "image_segmentation"], None, 3, None, config_2),
        (
            ["image_classification"],
            {"list_logger": {"path": "tests/deepsparse/loggers/helpers.py:ListLogger"}},
            2,
            "folder",
            config_3,
        ),
    ],
)
def test_data_logging_config_from_predefined(
    tmp_path, group_names, loggers, frequency, save_dir, expected_config
):
    config = data_logging_config_from_predefined(
        group_names=group_names, loggers=loggers, frequency=frequency, save_dir=save_dir
    )
    assert config == expected_config
    assert PipelineLoggingConfig(config)
    if save_dir:
        with open(os.path.join(tmp_path, save_dir, "data_logging_config.yaml")) as f:
            assert f.read() == expected_config
