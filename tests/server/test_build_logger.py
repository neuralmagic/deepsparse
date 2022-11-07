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
from deepsparse import FunctionLogger, PythonLogger
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig


yaml_config_1 = """
num_cores: 2
num_workers: 2
loggers:
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
             frequency: 5
           - func: tests/deepsparse/loggers/test_data/metric_functions.py:user_defined_identity
             frequency: 5
        - target: engine_outputs
          mappings:
           - func: np.mean
             frequency: 3"""

yaml_config_2 = """
num_cores: 2
num_workers: 2
loggers:
    - python
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


@pytest.mark.parametrize(
    "yaml_config, is_function_logger, is_logger",
    [
        (yaml_config_1, True, True),
        (yaml_config_2, False, True),
        (yaml_config_3, None, False),
    ],
)
def test_build_logger(yaml_config, is_function_logger, is_logger):
    obj = yaml.safe_load(yaml_config)
    server_config = ServerConfig(**obj)
    logger = build_logger(server_config)
    if is_logger:
        if is_function_logger:
            assert isinstance(logger, FunctionLogger)
            logger = logger.logger
        assert isinstance(logger, PythonLogger)
    else:
        assert logger is None
