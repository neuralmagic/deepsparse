# to be filled
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
import yaml

import pytest
from deepsparse.loggers import AsyncLogger, MultiLogger, PythonLogger
from deepsparse.server.build_logger import (
    build_logger,
    build_system_loggers,
    default_logger,
)
from deepsparse.server.config import ServerConfig, SystemLoggingConfig
from tests.deepsparse.loggers.helpers import ListLogger
from tests.helpers import find_free_port


yaml_config_1 = """
num_cores: 2
num_workers: 2
loggers:
    python:
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        pipeline_outputs:
           - func: identity
             frequency: 5
           - func: tests/test_data/metric_functions.py:user_defined_identity
             frequency: 5
        engine_outputs:
           - func: np.mean
             frequency: 3"""  # noqa E501

yaml_config_2 = """
num_cores: 2
num_workers: 2
loggers:
    python: {}
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1"""

yaml_config_3 = """
num_cores: 2
num_workers: 2
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1"""

yaml_config_4 = """
num_cores: 2
num_workers: 2
loggers:
    python:
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        pipeline_outputs:
           - func: tests/test_data/metric_functions.py:user_defined_identity
             frequency: 2
             target_loggers:
                - python
        engine_outputs:
           - func: np.mean
             frequency: 3"""

yaml_config_5 = """
num_cores: 2
num_workers: 2
loggers:
    invalid_logger_name:
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1"""

yaml_config_6 = """
num_cores: 2
num_workers: 2
loggers:
    python:
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        re:*_outputs:
          - func: tests/test_data/metric_functions.py:user_defined_identity
            frequency: 2"""

yaml_config_7 = """
num_cores: 2
num_workers: 2
loggers:
    python:
    prometheus:
        port: {port}
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        re:*_outputs:
          - func: tests/test_data/metric_functions.py:user_defined_identity
            frequency: 2"""

yaml_config_8 = """
num_cores: 2
num_workers: 2
loggers:
    custom_logger:
        path: tests/deepsparse/loggers/helpers.py:CustomLogger
        arg1: 1
        arg2: some_string
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        engine_outputs:
           - func: np.mean
             frequency: 3"""


@pytest.mark.parametrize(
    "yaml_config, raises_error, default_logger, num_function_loggers",
    [
        (yaml_config_1, False, False, 3),
        (yaml_config_2, False, False, 0),
        (yaml_config_3, False, True, None),
        (yaml_config_4, False, False, 2),
        (yaml_config_5, True, None, None),
        (yaml_config_6, False, False, 1),
        (yaml_config_7.format(port=find_free_port()), False, False, 1),
        (yaml_config_8, False, False, 1),
    ],
)
def test_build_logger(yaml_config, raises_error, default_logger, num_function_loggers):
    obj = yaml.safe_load(yaml_config)
    server_config = ServerConfig(**obj)
    if raises_error:
        with pytest.raises(ValueError):
            build_logger(server_config)
        return
    logger = build_logger(server_config)
    assert isinstance(logger, AsyncLogger)
    assert isinstance(logger.logger, MultiLogger)
    if default_logger:
        assert isinstance(logger.logger.loggers[0].logger.loggers[0], PythonLogger)
        return
    assert len(logger.logger.loggers) == num_function_loggers + 1

    # check for default system logger behaviour
    system_logger = logger.logger.loggers[-1]
    assert system_logger.target_identifier == "prediction_latency"
    assert system_logger.function_name == "identity"
    assert system_logger.frequency == 1


yaml_config_1 = """
system_logging: {}"""

yaml_config_2 = """
system_logging:
    enable: false"""

yaml_config_3 = """
system_logging:
    resource_utilization:
        enable: true"""

yaml_config_4 = """
system_logging:
    enable: false
    prediction_latency:
        enable: true
    resource_utilization:
        enable: true"""

yaml_config_5 = """
system_logging:
    prediction_latency:
        enable: true
    resource_utilization:
        enable: true
        target_loggers:
        - list_logger_1"""


@pytest.mark.parametrize(
    "yaml_config, expected_target_identifiers, number_leaf_loggers_per_system_logger",  # noqa: E501
    [
        (yaml_config_1, {"prediction_latency"}, [2]),
        (yaml_config_2, set(), []),
        (
            yaml_config_3,
            {
                "prediction_latency",
                "resource_utilization",
            },
            [2, 2],
        ),
        (yaml_config_4, set(), []),
        (
            yaml_config_5,
            {
                "prediction_latency",
                "resource_utilization",
            },
            [1, 2],
        ),
    ],
)
def test_build_system_loggers(
    yaml_config,
    expected_target_identifiers,
    number_leaf_loggers_per_system_logger,
):
    leaf_loggers = {"list_logger_1": ListLogger(), "list_logger_2": ListLogger()}
    obj = yaml.safe_load(yaml_config)
    system_logging_config = SystemLoggingConfig(**obj["system_logging"])
    system_loggers = build_system_loggers(leaf_loggers, system_logging_config)

    assert (
        set([logger.target_identifier for logger in system_loggers])
        == expected_target_identifiers
    )
    assert [
        len(system_logger.logger.loggers) for system_logger in system_loggers
    ] == number_leaf_loggers_per_system_logger


def test_default_logger(tmp_path):
    assert isinstance(default_logger()["python"], PythonLogger)
