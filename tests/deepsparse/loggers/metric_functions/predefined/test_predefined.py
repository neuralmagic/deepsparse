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
import pathlib
import time
from typing import Optional

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.loggers.build_logger import logger_from_config
from tests.deepsparse.loggers.helpers import fetch_leaf_logger
from tests.utils import mock_engine


def _generate_logs_path(group_name: str, optional_index: Optional[int] = None):
    logs_directory = os.path.join(
        pathlib.Path(__file__).parent.resolve(), "predefined_logs"
    )
    group_name = (
        f"{group_name}_{optional_index}" if optional_index is not None else group_name
    )
    return os.path.join(logs_directory, f"{group_name}.txt")


@pytest.mark.parametrize(
    "group_name, pipeline_name, inputs, optional_index",
    [
        (
            "image_classification",
            "image_classification",
            {"images": [numpy.ones((3, 224, 224))] * 2},
            None,
        ),
        (
            "image_classification",
            "image_classification",
            {"images": numpy.ones((2, 3, 224, 224))},
            None,
        ),
        ("object_detection", "yolo", {"images": [numpy.ones((3, 640, 640))] * 2}, None),
        ("object_detection", "yolo", {"images": numpy.ones((2, 3, 640, 640))}, None),
        ("segmentation", "yolact", {"images": [numpy.ones((3, 640, 640))] * 2}, None),
        ("segmentation", "yolact", {"images": numpy.ones((2, 3, 640, 640))}, None),
        (
            "question_answering",
            "question_answering",
            {
                "question": "what is the capital of France?",
                "context": "Paris is the capital of France.",
            },
            None,
        ),
        (
            "text_classification",
            "text_classification",
            {"sequences": [["Fun for adults and children.", "Fun for only children."]]},
            None,
        ),
        (
            "sentiment_analysis",
            "sentiment_analysis",
            {"sequences": "the food tastes great"},
            None,
        ),
        (
            "sentiment_analysis",
            "sentiment_analysis",
            {"sequences": ["the food tastes great", "the food tastes bad"]},
            1,
        ),
        (
            "sentiment_analysis",
            "sentiment_analysis",
            {
                "sequences": [
                    ["the food tastes great"],
                    ["the food tastes bad"],
                ]
            },
            2,
        ),
        (
            "zero_shot_text_classification",
            "zero_shot_text_classification",
            {"sequences": "the food tastes great", "labels": ["politics", "food"]},
            None,
        ),
        (
            "zero_shot_text_classification",
            "zero_shot_text_classification",
            {
                "sequences": ["the food tastes great", "the government is corrupt"],
                "labels": ["politics", "food"],
            },
            1,
        ),
        (
            "token_classification",
            "token_classification",
            {"inputs": "the food tastes great"},
            None,
        ),
        (
            "token_classification",
            "token_classification",
            {"inputs": ["the food tastes great", "the food tastes bad"]},
            1,
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_group_name(mock_engine, group_name, pipeline_name, inputs, optional_index):
    yaml_config = """
    loggers:
        list_logger:
            path: tests/deepsparse/loggers/helpers.py:ListLogger
    data_logging:
        predefined:
        - func: {group_name}"""

    if pipeline_name == "zero_shot_text_classification":
        pipeline = Pipeline.create(
            pipeline_name,
            logger=logger_from_config(
                config=yaml_config.format(group_name=group_name),
                pipeline_identifier=pipeline_name,
            ),
        )
    else:
        pipeline = Pipeline.create(
            pipeline_name, logger=yaml_config.format(group_name=group_name)
        )

    pipeline(**inputs)
    time.sleep(0.1)
    calls = fetch_leaf_logger(pipeline.logger).calls

    expected_logs = _generate_logs_path(group_name, optional_index)
    data_logging_logs = [call for call in calls if call.endswith("DATA")]
    if os.environ.get("NM_GENERATE_LOG_TEST_FILES"):
        dir = os.path.dirname(expected_logs)
        os.makedirs(dir, exist_ok=True)
        with open(expected_logs, "w") as f:
            f.write("\n".join(data_logging_logs))

    with open(expected_logs, "r") as f:
        expected_logs = f.read().splitlines()
    for log, expected_log in zip(data_logging_logs, expected_logs):
        assert log == expected_log


yaml_config = """
loggers:
    python:
data_logging:
    predefined:
    - func: image_classification
      frequency: 2
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
