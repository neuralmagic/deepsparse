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


from unittest import mock

from deepsparse.loggers import ManagerLogger, PrometheusLogger
from deepsparse.server.helpers import logger_manager_from_config


@mock.patch.object(PrometheusLogger, "_setup_client", lambda _: None)
def test_logger_manager_from_config(tmp_path):
    port = 8001
    text_log_save_dir = "/home/deepsparse-server/prometheus"
    text_log_save_freq = 30

    yaml_str = f"""
    prometheus:
        port: {port}
        text_log_save_dir: {text_log_save_dir}
        text_log_save_freq: {text_log_save_freq}
    """
    config_path = tmp_path / "loggers.yaml"
    with open(config_path, "w") as config_writer:
        config_writer.write(yaml_str)

    logger_manager = logger_manager_from_config(str(config_path))
    assert isinstance(logger_manager, ManagerLogger)

    loggers = logger_manager.loggers
    assert len(loggers) == 1
    assert "prometheus" in loggers
    logger = loggers["prometheus"]
    assert isinstance(logger, PrometheusLogger)
    assert logger.port == port
    assert logger.text_log_save_dir == text_log_save_dir
    assert logger.text_log_save_freq == text_log_save_freq
