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
from deepsparse import (
    AsyncLogger,
    MultiLogger,
    PythonLogger,
    default_logger,
    logger_from_config,
)
from deepsparse.loggers.build_logger import build_logger, build_system_loggers
from deepsparse.loggers.config import MetricFunctionConfig, SystemLoggingConfig
from tests.deepsparse.loggers.helpers import ListLogger, fetch_leaf_logger
from tests.helpers import find_free_port
from tests.utils import mock_engine


yaml_config_1 = """
loggers:
    python:
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
loggers:
    python: {}"""

yaml_config_3 = """
loggers:
    python:
data_logging:
    pipeline_outputs:
       - func: tests/test_data/metric_functions.py:user_defined_identity
         frequency: 2
         target_loggers:
            - python
    engine_outputs:
       - func: np.mean
         frequency: 3"""

yaml_config_4 = """
loggers:
    invalid_logger_name:
    """

yaml_config_5 = """
loggers:
    python:
data_logging:
    re:*_outputs:
      - func: tests/test_data/metric_functions.py:user_defined_identity
        frequency: 2"""

yaml_config_6 = """
loggers:
    python:
    prometheus:
        port: {port}
data_logging:
    re:*_outputs:
      - func: tests/test_data/metric_functions.py:user_defined_identity
        frequency: 2"""

yaml_config_7 = """
loggers:
    custom_logger:
        path: tests/deepsparse/loggers/helpers.py:CustomLogger
        arg1: 1
        arg2: some_string
data_logging:
    engine_outputs:
       - func: np.mean
         frequency: 3"""


@pytest.mark.parametrize(
    "yaml_config, raises_error, default_logger, num_function_loggers",
    [
        (yaml_config_1, False, False, 3),
        (yaml_config_2, False, False, 0),
        (yaml_config_3, False, False, 2),
        (yaml_config_4, True, None, None),
        (yaml_config_5, False, False, 1),
        (yaml_config_6.format(port=find_free_port()), False, False, 1),
        (yaml_config_7, False, False, 1),
    ],
)
@mock_engine(rng_seed=0)
def test_logger_from_config(
    engine_mock, yaml_config, raises_error, default_logger, num_function_loggers
):
    if raises_error:
        with pytest.raises(ValueError):
            logger_from_config(yaml_config)
        return
    logger = logger_from_config(yaml_config)

    assert isinstance(logger, AsyncLogger)
    assert isinstance(logger.logger, MultiLogger)
    if default_logger:
        assert isinstance(fetch_leaf_logger(logger), PythonLogger)
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


def test_kwargs():
    logger = build_logger(
        data_logging_config={"identifier": [MetricFunctionConfig(func="identity")]},
        system_logging_config=SystemLoggingConfig(),
        loggers_config={
            "kwargs_logger": {
                "path": "tests/deepsparse/loggers/helpers.py:KwargsLogger"
            }
        },
    )
    logger.log("identifier", None, None, argument="some_value")
    kwargs_logger = fetch_leaf_logger(logger)
    assert kwargs_logger.caught_kwargs == {"argument": "some_value"}
