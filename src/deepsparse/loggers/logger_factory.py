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

from deepsparse.loggers.registry.loggers.base_logger import BaseLogger
from deepsparse.loggers.root_logger import (
    LogType,
    MetricLogger,
    PerformanceLogger,
    SystemLogger,
)
from deepsparse.loggers.utils import import_from_path, import_from_registry


ROOT_LOGGER_DICT = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}


class LoggerFactory:
    """
    Factory to obtain root logger entrypoints given config file

    self.leaf_logger         # dict{key=logger_id, value=instantiated logger obj}
    self.root_logger_factory # dict{key=str, value=RootLogger}
    self.logger              # dict{key=LOG_TYPE.enum, value=RootLogger}

    """

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
    ) -> None:
        """
        Build the leaf logegr singleton

        Notes:
         name is the uuid of the logger, ex. default for
         PythonLogger (specified by the user)

        """
        logger_config = self.config.get("loggers")
        for name, init_args in logger_config.items():
            self.leaf_logger[name] = self.instantiate_logger(
                name=init_args.pop("name"),
                init_args=init_args,
            )

    def build_root_logger(self) -> None:
        """
        Build the root logger factory instantiating the
        root loggers with the leaf logger singleton and
        its section of the config file

        """

        for log_type, logger in ROOT_LOGGER_DICT.items():
            log_type_args = self.config.get(log_type)
            if log_type_args is not None:
                self.root_logger_factory[log_type] = logger(
                    config=self.config[log_type],
                    leaf_logger=self.leaf_logger,
                )

    def create(self) -> None:
        """Create the entrypoints to access the root loggers"""

        self.logger = {
            LogType.SYSTEM: self.root_logger_factory.get("system"),
            LogType.PERFORMANCE: self.root_logger_factory.get("performance"),
            LogType.METRIC: self.root_logger_factory.get("metric"),
        }

    def instantiate_logger(
        self, name: str, init_args: Dict[str, Any] = {}
    ) -> BaseLogger:
        """
        Instiate the logger from `name`, a path or the name of BaseLogger
        in the registry. Path example: path/to/file.py:LoggerName

        """
        if ":" in name:
            # Path example: path/to/file.py:LoggerName
            logger = import_from_path(path=name)
            return logger(**init_args)

        logger = import_from_registry(name)
        return logger(**init_args)
