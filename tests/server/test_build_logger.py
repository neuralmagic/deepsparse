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
from deepsparse import FunctionLogger, MultiLogger
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig


yaml_config_1 = """
num_cores: 2
num_workers: 2
loggers:
    python: {}
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        - target: pipeline_outputs
          functions:
           - func: identity
             frequency: 5
           - func: tests/test_data/metric_functions.py:user_defined_identity
             frequency: 5
        - target: engine_outputs
          functions:
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
    python: {}
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none
      batch_size: 1
      data_logging:
        - target: pipeline_outputs
          functions:
           - func: tests/test_data/metric_functions.py:user_defined_identity
             frequency: 2
             target_loggers: python
        - target: engine_outputs
          functions:
           - func: np.mean
             frequency: 3"""


@pytest.mark.parametrize(
    "yaml_config, returns_logger, is_top_logger_multilogger, num_function_loggers",
    [
        (yaml_config_1, True, True, 3),
        (yaml_config_2, True, False, 0),
        (yaml_config_3, False, None, None),
        (yaml_config_4, True, True, 2),
    ],
)
def test_build_logger(
    yaml_config, returns_logger, is_top_logger_multilogger, num_function_loggers
):
    obj = yaml.safe_load(yaml_config)
    server_config = ServerConfig(**obj)
    logger = build_logger(server_config)
    assert bool(logger) == returns_logger
    if not returns_logger:
        return

    assert is_top_logger_multilogger == isinstance(logger, MultiLogger)

    if num_function_loggers:
        assert len(logger.loggers) == num_function_loggers
        return
    assert not isinstance(logger, FunctionLogger)
