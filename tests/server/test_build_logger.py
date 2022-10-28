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
from deepsparse.loggers.config import (
    MultiplePipelinesLoggingConfig,
    PipelineLoggingConfig,
)
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig


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
             frequency: 2
           - func: tests/server/server_data/metric_functions.py:user_defined_identity
             frequency: 3
        - target: engine_outputs
          mappings:
           - func: np.mean
             frequency: 4"""

PIPELINE_CONFIG_1 = """
- name: question_answering
  targets:
    - target: pipeline_outputs
      mappings:
       - func: builtins:identity
         frequency: 2
       - func: tests/server/server_data/metric_functions.py:user_defined_identity
         frequency: 3
    - target: engine_outputs
      mappings:
       - func: np.mean
         frequency: 4"""

LOGGER_3 = logger_objects.FunctionLogger(
    logger=logger_objects.PythonLogger(),
    config=MultiplePipelinesLoggingConfig(
        pipelines=[
            PipelineLoggingConfig(**pipeline_config)
            for pipeline_config in yaml.safe_load(PIPELINE_CONFIG_1)
        ]
    ),
)

YAML_CONFIG_4 = """loggers:
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
             frequency: 2
           - func: tests/server/server_data/metric_functions.py:user_defined_identity
             frequency: 3
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      name: question_answering_
      data_logging:
        - target: engine_outputs
          mappings:
           - func: np.mean
             frequency: 4"""

PIPELINE_CONFIG_2 = """- name: question_answering
  targets:
    - target: pipeline_outputs
      mappings:
       - func: builtins:identity
         frequency: 2
       - func: tests/server/server_data/metric_functions.py:user_defined_identity
         frequency: 3
- name: question_answering_
  targets:
    - target: engine_outputs
      mappings:
       - func: np.mean
         frequency: 4"""

LOGGER_4 = logger_objects.FunctionLogger(
    logger=logger_objects.PythonLogger(),
    config=MultiplePipelinesLoggingConfig(
        pipelines=[
            PipelineLoggingConfig(**pipeline_config)
            for pipeline_config in yaml.safe_load(PIPELINE_CONFIG_2)
        ]
    ),
)


@pytest.mark.parametrize(
    "yaml_server_config,expected_logger",
    [
        (YAML_CONFIG_1, LOGGER_1),
        (YAML_CONFIG_2, LOGGER_2),
        (YAML_CONFIG_3, LOGGER_3),
        (YAML_CONFIG_4, LOGGER_4),
    ],
)
def test_build_logger(yaml_server_config, expected_logger):
    obj = yaml.safe_load(yaml_server_config)
    server_config = ServerConfig(**obj)
    logger = build_logger(server_config)
    _is_equal(logger, expected_logger)


def _is_equal(logger, expected_logger):
    # figure out a smart way to test here
    assert True
