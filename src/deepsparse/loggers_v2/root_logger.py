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


from collections import defaultdict
from enum import Enum
from typing import Any, Dict, Optional

from deepsparse.loggers_v2.filters.frequency_filter import FrequencyFilter
from deepsparse.loggers_v2.filters.pattern import is_match_found

from .utils import import_from_registry


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRIC = "METRIC"


class RootLogger(FrequencyFilter):
    def __init__(self, config: Dict, leaf_logger):
        super().__init__()
        self.config = config
        self.leaf_logger = leaf_logger
        self.logger = defaultdict(list)  # tag as key
        self.func = {}  # func name: callable
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
                self.func[func] = import_from_registry(func)
                super().add_template_to_frequency(
                    tag=tag, func=func, rate=func_arg.get("freq", 1)
                )
                for logger_id in func_arg.get("uses", []):
                    self.logger[tag].append(self.leaf_logger[logger_id])

    def log(
        self, value: Any, log_type: str, tag: Optional[str] = None, *args, **kwargs
    ):
        """
        Send args to its appropriate logger if the given tag is valid
        """
        for pattern, loggers in self.logger.items():
            if is_match_found(pattern, tag):
                for func_name, func_callable in self.func.items():
                    if super().should_execute_on_frequency(
                        tag=tag, log_type=log_type, func=func_name
                    ):
                        value = func_callable(value)
                        for logger in loggers:
                            logger.log(
                                value=value,
                                tag=tag,
                                func=func_name,
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

    def log(self, capture: str, *args, **kwargs):
        for pattern in self.capture:
            if is_match_found(pattern, capture):
                super().log(log_type=self.LOG_TYPE, capture=capture, *args, **kwargs)
