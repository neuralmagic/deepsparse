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

"""
Helper functions for deepsparse.server
"""


import logging

import yaml

from deepsparse.loggers import ManagerLogger, PrometheusLogger


__all__ = [
    "default_logger_manager",
    "logger_manager_from_config",
]


_LOGGER = logging.getLogger(__name__)
_SUPPORTED_LOGGERS = {"prometheus": PrometheusLogger}


def default_logger_manager() -> ManagerLogger:
    """
    :return: default ManagerLogger object for the deployment scenario
    """
    # always return prometheus logger for now
    return ManagerLogger(PrometheusLogger())


def logger_manager_from_config(config_path: str) -> ManagerLogger:
    """
    Initializes a ManagerLogger from loggers defined by a yaml config file

    Config should be a logger integration name, followed by its initialization
    kwargs as a dict

    supported logger integrations:
    ["prometheus"]

    example config:

    ```yaml
    prometheus:
        port: 8001
        text_log_save_dir: /home/deepsparse-server/prometheus
        text_log_save_freq: 30
    ```

    :param config_path: path to YAML config file
    :return: constructed ManagerLogger object
    """
    with open(config_path) as config:
        loggers_config = yaml.safe_load(config)

    if not isinstance(loggers_config, dict):
        raise ValueError(
            "Loggers config must be a yaml dict, loaded object of "
            f"type {type(loggers_config)}"
        )

    loggers = []

    for integration, logger_kwargs in loggers_config.items():
        logger_class = _SUPPORTED_LOGGERS.get(integration.lower())
        if logger_class is None:
            raise ValueError(
                f"Unknown logger integration {integration}. Supported integrations: "
                f"{list(_SUPPORTED_LOGGERS)}"
            )
        logger = logger_class(**logger_kwargs)
        _LOGGER.info(f"Created logger {logger}")
        loggers.append(logger)

    return ManagerLogger(loggers=loggers)
