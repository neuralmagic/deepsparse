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

from .async_logger import AsyncLogger


class LoggerType(Enum):
    STREAM = logging.StreamHandler
    FILE = logging.FileHandler
    ROTATING = RotatingFileHandler


def create_file_if_not_exists(filename):
    if not os.path.exists(filename):
        open(filename, "a").close()


class SystemLoggerFactory:
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


import importlib


def get_logger_from_path(path: str):
    breakpoint()
    path, class_name = path.split(":")
    path = path.split(".py")[0]

    _path = path
    path = path.replace(r"/", ".")
    try:
        module = importlib.import_module(path)
    except:
        raise ValueError(f"Cannot find module with path {_path}")
    try:
        return getattr(module, class_name)
    except:
        raise ValueError(f"Cannot find Class with name {class_name} in {_path}")


from typing import Any, Optional


class PerformanceLoggerFactory(AsyncLogger):
    """
    Initialize all loggers from config. Log to all loggers using log

    :param log: Callable that calls all the loggers
    """

    def __init__(self, config):
        super().__init__(**config)
        self.logger = []

    def create_logger(self):
        for logger_path, config in self.config.items():
            logger = get_logger_from_path(logger_path)
            config = {}
            breakpoint()
            self.logger.append(logger(**config))

    def log(self, value: Any, tag: Optional[str] = None, *args, **kwargs):
        for logger in self.logger:
            super().log(
                logger=logger,
                value=value,
                tag=tag,
                identifier=1,
                category=1,
                *args,
                **kwargs,
            )
            breakpoint()
