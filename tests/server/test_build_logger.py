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
from deepsparse.loggers import MetricCategories
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig
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


@pytest.mark.parametrize(
    "yaml_config, raises_error, returns_logger, num_function_loggers",
    [
        (yaml_config_1, False, True, 3),
        (yaml_config_2, False, True, 0),
        (yaml_config_3, False, False, None),
        (yaml_config_4, False, True, 2),
        (yaml_config_5, True, None, None),
        (yaml_config_6, False, True, 1),
        (yaml_config_7.format(port=find_free_port()), False, True, 1),
    ],
)
def test_build_logger(yaml_config, raises_error, returns_logger, num_function_loggers):
    obj = yaml.safe_load(yaml_config)
    server_config = ServerConfig(**obj)
    if raises_error:
        with pytest.raises(ValueError):
            build_logger(server_config)
        return
    logger = build_logger(server_config)
    assert bool(logger) == returns_logger
    if not returns_logger:
        return
    assert len(logger.loggers) == num_function_loggers + 1
    # check for system logger
    system_logger = logger.loggers[-1]
    assert system_logger.target_identifier == (
        f"category:{MetricCategories.SYSTEM.value}"
    )
    assert system_logger.function_name == "identity"
    assert system_logger.frequency == 1
