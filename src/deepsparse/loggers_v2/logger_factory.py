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
from typing import Any, Dict


# from deepsparse.loggers import custom_logger_from_identifier

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


def import_from_registry(name: str):
    registry = "src.deepsparse.loggers_v2.registry.__init__"
    module = importlib.import_module(registry)

    try:
        return getattr(module, name)
    except:
        raise ValueError(f"Cannot find Class with name {name} in {registry}")


import re
from collections import defaultdict
from typing import Callable, Optional


def should_allow_log_by_tag(
    pattern: str,
    tag: Optional[str] = None,
):
    if pattern == "*":
        return True
    if tag is not None and re.match(pattern, tag):
        return True
    return False


class RootLogger:
    DEFAULT_LOGGER_MAP = {
        "default": "DefaultLogger",
        "promoetheus": "PrometheusLogger",
    }

    def __init__(self, config: Dict):
        self.config = config
        self.logger = defaultdict(list)
        self.func = set()
        self.tag = set()

        self.create()

    def create(self):
        """
        Instantiate the loggers as singleton and 
            import the class/func from registry
        """
        for logger_name, init_args in self.config.items():
            logger = import_from_registry(
                self.DEFAULT_LOGGER_MAP.get(logger_name, logger_name)
            )

            logger_singleton = logger(
                frequency=init_args.get("frequency"),
                handler=init_args.get("handler"),
            )

            for func_name in init_args.get("func"):
                fn = import_from_registry(func_name)
                self.func.add(fn)

            for tag in init_args["tag"]:
                self.logger[tag].append(logger_singleton)
                self.tag.add(tag)

    def log(self, value: Any, tag: Optional[str] = None, *args, **kwargs):
        """
        Send args to its appropriate logger if the given tag is valid
        """
        for pattern, loggers in self.logger.items():
            if should_allow_log_by_tag(pattern, tag):
                for logger in loggers:
                    for func in self.func:
                        print(logger)
                        logger.log(value=value, tag=tag, func=func, *args, **kwargs)


class SystemLogger(RootLogger):
    """
    Create Python level logging with handles
    """

    ...


class PerformanceLogger(RootLogger):
    """
    Create performance level (in-line pipeline)
        logging with handles
    """

    ...


class MetricLogger(RootLogger):
    """
    Create metric level (logged in LoggerMiddleware)
        logging with handles
    """

    ...


ROOT_LOGGERS = {
    "system": SystemLogger,
    "performance": PerformanceLogger,
    "metric": MetricLogger,
}


def logger_factory(config: Dict) -> Dict[str, RootLogger]:

    loggers = {}
    for log_type, logger in ROOT_LOGGERS.items():
        log_type_args = config.get(log_type)
        if log_type_args is not None:
            loggers[log_type] = logger(
                config=config[log_type],
            )
    return loggers
