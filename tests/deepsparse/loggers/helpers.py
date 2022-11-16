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
Helper classes and functions for testing deepsparse.loggers
"""

import os
from datetime import datetime
from time import sleep
from typing import Any

from deepsparse.loggers import BaseLogger, MetricCategories


__all__ = [
    "ErrorLogger",
    "FileLogger",
    "NullLogger",
    "SleepLogger",
]


class ErrorLogger(BaseLogger):
    # raises an error on log for testing purposes

    def log(self, identifier: str, value: Any, category: MetricCategories):
        raise RuntimeError("Raising for testing purposes")


class FileLogger(BaseLogger):
    # NOT THREAD SAFE - should be used for testing accordingly
    # logs results by appending to the given file_path

    def __init__(self, file_path: str):
        self.file_path = file_path

        # create file
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w"):
                pass

    def log(self, identifier: str, value: Any, category: MetricCategories):
        msg = (
            f" Identifier: {identifier} | Category: {category.value} "
            f"| Logged Data: {value}"
        )
        msg = datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f") + msg

        with open(self.file_path, "a") as file:
            file.write(msg + "\n")


class NullLogger(BaseLogger):
    # leaf logger that does nothing

    def log(self, identifier: str, value: Any, category: MetricCategories):
        pass


class SleepLogger(BaseLogger):
    # sleeps thread for sleep_time before forwarding to wrapped logger

    def __init__(self, logger: BaseLogger, sleep_time: int = 1):
        self.logger = logger
        self.sleep_time = sleep_time

    def log(self, identifier: str, value: Any, category: MetricCategories):
        sleep(self.sleep_time)
        self.logger.log(identifier=identifier, value=value, category=category)


class CustomLogger(BaseLogger):
    # mock custom logger for testing purposes

    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def log(self, identifier: str, value: Any, category: MetricCategories):
        pass


class ListLogger(BaseLogger):
    # leaf logger that aggregates its log in a list
    def __init__(self):
        self.calls = []

    def log(self, identifier, value, category):
        self.calls.append(
            f"identifier:{identifier}, value:{value}, category:{category}"
        )
