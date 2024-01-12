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

import importlib
import re
from collections import defaultdict

# from deepsparse.loggers import custom_logger_from_identifier
from threading import Lock
from typing import Any, Callable, Dict, Optional


class FrequncyExecutor:
    def __init__(self):
        self._lock = Lock()
        self.frequency = {}
        self.counter = {}

    def should_execute_on_frequency(
        self, value: Any, tag: str, log_type: str, func: Callable
    ) -> bool:
        # stub = f"{log_type}.{tag}.{func}.{value}"
        stub = f"{log_type}.{tag}.{func}"

        stub_frequency = f"{tag}.{func}"
        with self._lock:
            if stub not in self.counter:
                self.counter[stub] = 0
            frequency = self.frequency.get(stub_frequency)
            self.counter[stub] = (self.counter[stub] + 1) % frequency
            should_execute = self.counter[stub] == 0
            print(self.counter[stub], stub)
        if should_execute:
            return True
        return False


def import_from_registry(name: str):
    registry = "src.deepsparse.loggers_v2.registry.__init__"
    module = importlib.import_module(registry)
    try:
        return getattr(module, name)
    except:
        raise ValueError(f"Cannot find Class with name {name} in {registry}")


def import_from_path(path: str):
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
        raise ValueError(f"Cannot find {class_name} in {_path}")


def should_allow_log_by_pattern(
    pattern: str,
    string: Optional[str] = None,
):
    if pattern == "*":
        return True
    if string is not None and re.match(pattern, string):
        return True
    return False


def instantiate_logger(name: str, init_args: Dict[str, Any] = {}):
    if ":" in name:
        # path/to/file.py:class_or_func
        logger = import_from_path(path=name)
        return logger(**init_args)

    logger = import_from_registry(name)
    return logger(**init_args)


class RootLogger(FrequncyExecutor):
    # DEFAULT_LOGGER_MAP = {
    #     "default": "DefaultLogger",
    #     "promoetheus": "PrometheusLogger",
    # }

    def __init__(self, config: Dict, leaf_logger):
        super().__init__()
        self.config = config
        self.leaf_logger = leaf_logger
        self.logger = defaultdict(list)
        self.func = set()
        self.tag = set()
        self.create()

    def create(self):
        """
        Instantiate the loggers as singleton and
            import the class/func from registry
        """
        for tag, func_args in self.config.items():
            for func_arg in func_args:
                func = func_arg.get("func")
                self.func.add(func)
                self.frequency[f"{tag}.{func}"] = func_arg.get("freq", 1)
                for logger_id in func_arg.get("uses", []):
                    self.logger[tag].append(self.leaf_logger[logger_id])

    def log(
        self, value: Any, log_type: str, tag: Optional[str] = None, *args, **kwargs
    ):
        """
        Send args to its appropriate logger if the given tag is valid
        """
        for pattern, loggers in self.logger.items():
            if should_allow_log_by_pattern(pattern, tag):
                for func in self.func:
                    if super().should_execute_on_frequency(
                        value=value, tag=tag, log_type=log_type, func=func
                    ):
                        for logger in loggers:
                            logger.log(
                                value=value,
                                tag=tag,
                                func=func,
                                log_type=log_type,
                                *args,
                                **kwargs,
                            )


class SystemLogger(RootLogger):
    """
    Create Python level logging with handles
    """

    LOG_TYPE = "system"

    def log(self, *args, **kwargs):
        super().log(log_type=self.LOG_TYPE, *args, **kwargs)


class PerformanceLogger(RootLogger):
    """
    Create performance level (in-line pipeline)
        logging with handles
    """

    LOG_TYPE = "performance"

    def log(self, *args, **kwargs):
        super().log(log_type=self.LOG_TYPE, *args, **kwargs)


class MetricLogger(RootLogger):
    """
    Create metric level (logged in LoggerMiddleware)
        logging with handles
    """

    LOG_TYPE = "metric"

    def __init__(self, config: Dict, leaf_logger):
        self.capture = set()
        super().__init__(config, leaf_logger)

    def create(self):
        super().create()
        for func_args in self.config.values():
            for func_arg in func_args:
                for capture in func_arg["capture"]:
                    self.capture.add(capture)

    # def log(self, value: Any, tag: Optional[str] = None, *args, **kwargs):
    #     if isinstance(value, Dict):
    #         for key, val in value.items():
    #             for pattern in self.capture:
    #                 if should_allow_log_by_pattern(pattern, key):
    #                     super().log(
    #                         value=val, tag=tag, log_type=self.LOG_TYPE, *args, **kwargs
    #                     )


ROOT_LOGGER = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}


def logger_factory(config: Dict, leaf_logger: Dict) -> Dict[str, RootLogger]:
    loggers = {}
    for log_type, logger in ROOT_LOGGER.items():
        log_type_args = config.get(log_type)
        if log_type_args is not None:
            # breakpoint()
            loggers[log_type] = logger(
                config=config[log_type],
                leaf_logger=leaf_logger,
            )
    return loggers
