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


class LoggerType(Enum):
    STREAM = logging.StreamHandler
    FILE = logging.FileHandler
    ROTATING = RotatingFileHandler


def create_file_if_not_exists(filename):
    if not os.path.exists(filename):
        open(filename, "a").close()


class LoggingConfigFactory:
    def __init__(self, config):

        self.config = config
        self.logger = logging.getLogger()  # Use the root logger
        self.logger.setLevel(config.pop("level", "info"))

    def create_logger(self):
        for handler_type, handler_config in self.config.items():
            level = handler_config.pop("level", "INFO")
            handler = self.create_handler(handler_type, handler_config)
            handler.setLevel(level)
            self.logger.addHandler(handler)
        return self.logger

    def create_handler(self, handler_type, handler_config):
        logger_class = LoggerType[handler_type.upper()].value

        # Set handler level
        handler_level = handler_config.pop("level", logging.INFO)

        if handler_type == "stream":
            handler = logger_class()
        elif handler_type == "file":
            filename = handler_config.get("filename", "")
            create_file_if_not_exists(filename)
            handler = logger_class(filename=filename)
        elif handler_type == "rotating":
            filename = handler_config.get("filename", "")
            create_file_if_not_exists(filename)

            handler = logger_class(
                filename=filename,
                maxBytes=handler_config.get("max_bytes", 0),
                backupCount=handler_config.get("backup_count", 0),
            )
        else:
            raise ValueError(f"Unsupported logger type: {handler_type}")

        handler.setLevel(handler_level)

        # Set handler formatter
        formatter = logging.Formatter(handler_config.get("formatter", ""))
        handler.setFormatter(formatter)

        return handler
