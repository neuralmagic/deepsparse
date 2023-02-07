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
import time

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.loggers.build_logger import logger_from_config
from tests.utils import mock_engine
from tests.deepsparse.loggers.helpers import fetch_leaf_logger


@pytest.mark.parametrize(
    "group_name, pipeline_name, inputs, expected_logs",
    [
        (
            "image_classification",
            "image_classification",
            {"images": [numpy.ones((3, 224, 224))] * 2},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/image_classification_logs.txt",  # noqa E501
        ),
        (
            "image_classification",
            "image_classification",
            {"images": numpy.ones((2, 3, 224, 224))},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/image_classification_logs.txt",  # noqa E501
        ),
        (
            "object_detection",
            "yolo",
            {"images": [numpy.ones((3, 640, 640))] * 2},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/object_detection_logs.txt",  # noqa E501
        ),
        (
            "object_detection",
            "yolo",
            {"images": numpy.ones((2, 3, 640, 640))},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/object_detection_logs.txt",  # noqa E501
        ),
        (
            "segmentation",
            "yolact",
            {"images": [numpy.ones((3, 640, 640))] * 2},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/segmentation_logs.txt",  # noqa E501
        ),
        (
            "segmentation",
            "yolact",
            {"images": numpy.ones((2, 3, 640, 640))},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/segmentation_logs.txt",  # noqa E501
        ),
        (
            "question_answering",
            "question_answering",
            {
                "question": "what is the capital of France?",
                "context": "Paris is the capital of France.",
            },
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/question_answering_logs.txt",  # noqa E501
        ),
        (
            "text_classification",
            "text_classification",
            {"sequences": [["Fun for adults and children.", "Fun for only children."]]},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/text_classification_logs.txt",  # noqa E501
        ),
        (
            "sentiment_analysis",
            "sentiment_analysis",
            {"sequences": "the food tastes great"},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/sentiment_analysis_logs_1.txt",  # noqa E501
        ),
        (
            "sentiment_analysis",
            "sentiment_analysis",
            {"sequences": ["the food tastes great", "the food tastes bad"]},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/sentiment_analysis_logs_2.txt",  # noqa E501
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
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/sentiment_analysis_logs_3.txt",  # noqa E501
        ),
        (
            "zero_shot_text_classification",
            "zero_shot_text_classification",
            {"sequences": "the food tastes great", "labels": ["politics", "food"]},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/zero_shot_text_classification_logs_1.txt",  # noqa E501
        ),
        (
            "zero_shot_text_classification",
            "zero_shot_text_classification",
            {
                "sequences": ["the food tastes great", "the government is corrupt"],
                "labels": ["politics", "food"],
            },
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/zero_shot_text_classification_logs_2.txt",  # noqa E501
        ),
        (
            "token_classification",
            "token_classification",
            {"inputs": "the food tastes great"},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/token_classification_logs_1.txt",  # noqa E501
        ),
        (
            "token_classification",
            "token_classification",
            {"inputs": ["the food tastes great", "the food tastes bad"]},
            "tests/deepsparse/loggers/metric_functions/template_functions/template_logs/token_classification_logs_2.txt",  # noqa E501
        ),
    ],
)
@mock_engine(rng_seed=0)
def test_group_name(mock_engine, group_name, pipeline_name, inputs, expected_logs):
    yaml_config = """
    loggers:
        list_logger:
            path: tests/deepsparse/loggers/helpers.py:ListLogger
    add_predefined:
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
    # let's sleep one seconds so all the logs get collected
    time.sleep(1)
    calls = fetch_leaf_logger(pipeline.logger).calls

    data_logging_logs = [call for call in calls if "DATA" in calls]
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
    from_predefined:
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
