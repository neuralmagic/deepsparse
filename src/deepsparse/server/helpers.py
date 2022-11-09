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

from deepsparse import BaseLogger, PythonLogger


__all__ = ["log_system_info", "default_logger"]

_LOGGER = logging.getLogger(__name__)


def default_logger() -> PythonLogger:
    """
    :return: default PythonLogger object for the deployment scenario
    """
    logger = PythonLogger()
    _LOGGER.info("Created default logger: PythonLogger")
    return logger


def log_system_info(server_logger: BaseLogger):
    pass
