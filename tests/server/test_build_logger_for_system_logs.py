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
from deepsparse.server.build_logger import build_logger
from deepsparse.server.config import ServerConfig


yaml_config_1 = """
num_cores: 2
num_workers: 2
system_logging:
    inference_latency_group:
        enable: true
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"""  # noqa E501

yaml_config_2 = """
num_cores: 2
num_workers: 2
system_logging:
    SHOULD_RAISE_ERROR_group:
        enable: true
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"""  # noqa E501

yaml_config_3 = """
num_cores: 2
num_workers: 2
system_logging:
    enable: false
    inference_latency_group:
        enable: true
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"""  # noqa E501

yaml_config_4 = """
num_cores: 2
num_workers: 2
endpoints:
    - task: question_answering
      route: /unpruned/predict
      model: zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none"""  # noqa E501


@pytest.mark.parametrize(
    "yaml_config, raises_error, expected_identifier",
    [
        (yaml_config_1, False, "category:system/inference_latency"),
        (yaml_config_2, True, None),
        (yaml_config_3, False, None),
        (yaml_config_4, False, "category:system/inference_latency"),
    ],
)
def test_build_logger(yaml_config, raises_error, expected_identifier):
    obj = yaml.safe_load(yaml_config)
    server_config = ServerConfig(**obj)
    if raises_error:
        with pytest.raises(ValueError):
            build_logger(server_config)
        return
    logger = build_logger(server_config)

    if expected_identifier is None:
        assert logger.logger.loggers == []
        return
    assert logger.logger.loggers[0].target_identifier == expected_identifier
