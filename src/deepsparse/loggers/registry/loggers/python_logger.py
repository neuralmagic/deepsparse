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


import logging
import os
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from deepsparse.loggers.registry.loggers.base_logger import BaseLogger


class LoggerType(Enum):
    STREAM = logging.StreamHandler
    FILE = logging.FileHandler
    ROTATING = RotatingFileHandler


def create_file_if_not_exists(filename):
    if not os.path.exists(filename):
        open(filename, "a").close()


class PythonLogger(BaseLogger):
    def __init__(
        self,
        handler: Optional[Dict] = None,
    ):
        self.handler = handler
        self.logger = logging.getLogger()  # Root loggger
        self.set_hander()

    def set_hander(self):
        ...

    def log(
        self,
        value: Any,
        tag: str,
        log_type: str,
        func: Optional[str] = None,
        level: str = "info",
        **kwargs,
    ):
        placeholders = f"[{log_type}.{tag}.{str(func)}]"
        if (run_time := kwargs.get("time")) is not None:
            placeholders += f"[⏱️{run_time}]"

        logger = getattr(self.logger, level)
        logger(f"{placeholders}: {value}")


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Add your custom placeholders to the log record
        record.placeholders = f"[{record.log_type}.{record.tag}.{str(record.func)}]"
        if hasattr(record, "run_time"):
            record.placeholders += f"[⏱️{record.run_time}]"

        # Use the original formatter to format the log message
        return super().format(record)
