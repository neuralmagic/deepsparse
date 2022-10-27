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
from deepsparse import loggers as logger_objects
from deepsparse.loggers.configs import PipelineLoggingConfig
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig


_ = """num_workers: 2
loggers:
    - prometheus:
         port: 8001
    - python
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:cv/...
      batch_size: 1
      data_logging:
         pipeline_outputs:
           - func: builtins:batch-mean
             frequency: 50
           - func: ./estimator_file.py:my_func_name
             frequency: 50
         engine_outputs:
           - func: np.mean
"""

DATA_LOGGING = """- target: pipeline_outputs
          mapping:
           - func: builtins:identity
             frequency: 50
           - func: tests/server/server_data/metric_function.py:user_defined_identity
             frequency: 50
        - target: engine_outputs
          mapping:
           - func: np.mean
             frequency: 20"""

DATA_LOGGING_ = """
question_answering:
    target: pipeline_outputs
    mapping:
        - func: builtins:identity
          frequency: 50
          func_name: identity
        - func: tests/server/server_data/metric_function.py:user_defined_identity
          frequency: 50
          func_name: user_defined_identity
    target: engine_outputs
    mapping:
        - func: np.mean
          frequency: 20
          func_name: mean"""


YAML_CONFIG_1 = """endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
"""
LOGGER_1 = None

YAML_CONFIG_2 = """loggers:
    - python
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
"""
LOGGER_2 = logger_objects.PythonLogger()

YAML_CONFIG_3 = """loggers:
    - prometheus:
        port: 6001
    - python
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
"""
LOGGER_3 = logger_objects.MultiLogger(
    loggers=[logger_objects.PrometheusLogger(port=6001), logger_objects.PythonLogger()]
)


YAML_CONFIG_4 = """loggers:
    - prometheus:
        port: 6001
    - python
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        - target: pipeline_outputs
          mappings:
           - func: builtins:identity
             frequency: 50
           - func: tests/server/server_data/metric_function.py:user_defined_identity
             frequency: 50
        - target: engine_outputs
          mappings:
           - func: np.mean
             frequency: 20"""

PIPELINE_CONFIG_1 = """
- name: question_answering
  targets:
    - target: pipeline_outputs
      mappings:
       - func: builtins:identity
         frequency: 50
       - func: tests/server/server_data/metric_function.py:user_defined_identity
         frequency: 50
    - target: engine_outputs
      mappings:
       - func: np.mean
         frequency: 20"""

LOGGER_4 = logger_objects.FunctionLogger(
    logger=logger_objects.MultiLogger(
        loggers=[
            logger_objects.PrometheusLogger(port=6001),
            logger_objects.PythonLogger(),
        ]
    ),
    config=[
        PipelineLoggingConfig(**pipeline_config)
        for pipeline_config in yaml.safe_load(PIPELINE_CONFIG_1)
    ],
)

YAML_CONFIG_5 = """loggers:
    - prometheus:
        port: 6001
    - python
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        - target: pipeline_outputs
          mappings:
           - func: builtins:identity
             frequency: 50
           - func: tests/server/server_data/metric_function.py:user_defined_identity
             frequency: 50
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      name: {name}
      data_logging:
        - target: engine_outputs
          mappings:
           - func: np.mean
             frequency: 20"""

PIPELINE_CONFIG_2 = """- name: question_answering
  targets:
    - target: pipeline_outputs
      mappings:
       - func: builtins:identity
         frequency: 50
       - func: tests/server/server_data/metric_function.py:user_defined_identity
         frequency: 50
- name: question_answering_
  targets:
    - target: engine_outputs
      mappings:
       - func: np.mean
         frequency: 20"""

LOGGER_5 = logger_objects.FunctionLogger(
    logger=logger_objects.MultiLogger(
        loggers=[
            logger_objects.PrometheusLogger(port=6001),
            logger_objects.PythonLogger(),
        ]
    ),
    config=[
        PipelineLoggingConfig(**pipeline_config)
        for pipeline_config in yaml.safe_load(PIPELINE_CONFIG_2)
    ],
)


@pytest.mark.parametrize(
    "yaml_server_config,expected_logger, should_fail",
    [
        # (YAML_CONFIG_1, LOGGER_1, False),
        # (YAML_CONFIG_2, LOGGER_2, False),
        # (YAML_CONFIG_3, LOGGER_3, False),
        # (YAML_CONFIG_4, LOGGER_4, False),
        # (YAML_CONFIG_5.format(name="question_answering_"), LOGGER_5, False),
        (YAML_CONFIG_5.format(name="question_answering"), None, True),
    ],
)
def test_build_logger(yaml_server_config, expected_logger, should_fail):
    obj = yaml.safe_load(yaml_server_config)
    server_config = ServerConfig(**obj)
    logger = build_logger(server_config)
    _is_equal(logger, expected_logger)


def _is_equal(logger, expected_logger):

    if expected_logger is None:
        assert logger is None
        return
    elif isinstance(expected_logger, logger_objects.PythonLogger):
        assert isinstance(logger, logger_objects.PythonLogger)
        return
    elif isinstance(expected_logger, logger_objects.MultiLogger):
        for _expected_logger, _logger in zip(expected_logger.loggers, logger.loggers):
            assert type(_expected_logger) == type(_logger)
            assert _expected_logger.__dict__ == _logger.__dict__
    elif isinstance(expected_logger, logger_objects.FunctionLogger):
        pass
