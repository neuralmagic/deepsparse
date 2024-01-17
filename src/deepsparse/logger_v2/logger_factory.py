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


from typing import Any, Dict

from .root_logger import LogType, MetricLogger, PerformanceLogger, SystemLogger
from .utils import import_from_path, import_from_registry


ROOT_LOGGER_DICT = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}


class LoggerFactory:
    def __init__(self, config: Dict[str, Dict]):
        self.config = config

        self.leaf_logger = {}
        self.root_logger_factory = {}
        self.logger = {}

        self.build_leaf_logger()
        self.build_root_logger()

        self.create()

    def build_leaf_logger(
        self,
    ):
        """Build the leaf logegr singleton"""
        logger_config = self.config.get("logger")
        for name, init_args in logger_config.items():
            self.leaf_logger[name] = self.instantiate_logger(
                name=init_args.pop("name"),
                init_args=init_args,
            )

    def build_root_logger(self):

        for log_type, logger in ROOT_LOGGER_DICT.items():
            log_type_args = self.config.get(log_type)
            if log_type_args is not None:
                self.root_logger_factory[log_type] = logger(
                    config=self.config[log_type],
                    leaf_logger=self.leaf_logger,
                )

    def create(self):
        self.logger = {
            LogType.SYSTEM: self.root_logger_factory.get("system"),
            LogType.PERFORMANCE: self.root_logger_factory.get("performance"),
            LogType.METRIC: self.root_logger_factory.get("metric"),
        }

    def instantiate_logger(self, name: str, init_args: Dict[str, Any] = {}):
        if ":" in name:
            # path/to/file.py:class_or_func
            logger = import_from_path(path=name)
            return logger(**init_args)

        logger = import_from_registry(name)
        return logger(**init_args)
