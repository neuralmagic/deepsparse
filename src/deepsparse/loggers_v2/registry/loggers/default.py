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
from typing import Any, Callable, Dict, List, Optional

from .frequency_logger import FrequencyLogger


class LoggerType(Enum):
    STREAM = logging.StreamHandler
    FILE = logging.FileHandler
    ROTATING = RotatingFileHandler


def create_file_if_not_exists(filename):
    if not os.path.exists(filename):
        open(filename, "a").close()


# def get_logger_from_path(path: str):
#     breakpoint()
#     path, class_name = path.split(":")
#     path = path.split(".py")[0]

#     _path = path
#     path = path.replace(r"/", ".")
#     try:
#         module = importlib.import_module(path)
#     except:
#         raise ValueError(f"Cannot find module with path {_path}")
#     try:
#         return getattr(module, class_name)
#     except:
#         raise ValueError(f"Cannot find {class_name} in {_path}")


class DefaultLogger(FrequencyLogger):
    # class DefaultLogger:

    def __init__(
        self,
        frequency: int = 1,
        handler: Optional[Dict] = None,
    ):
        super().__init__(frequency)

        self.handler = handler
        self.logger = logging.getLogger()  # Root loggger
        self.set_hander()

    def set_hander(self):
        ...

    def log(
        self, value: Any, tag: str, func: Optional[Callable] = None, level: str = "info"
    ):
        logger = getattr(self.logger, level)
        super().log(logger=logger, value=value, tag=tag, func=func)
