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
from deepsparse.loggers_v2.filters.pattern import (
    is_match_found,
    unravel_value_as_generator,
)

from .utils import import_from_registry


class LogType(Enum):
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    METRIC = "METRIC"


class RootLogger(FrequencyFilter):
    """
    Child class for SystemLogger, PerformanceLogger, MetricLogger
    All class instantitated with RootLogger will have its own FrequencyFilter

    :param config: config with respect to the log_type (LoggerConfig().dict().get(log_type))
    :param leaf_logger: leaf logger singleton shared among other RootLogger

    """

    def __init__(self, config: Dict, leaf_logger: Dict):
        super().__init__()
        self.config = config
        self.leaf_logger = leaf_logger
        self.run_args = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.create()

    def create(self):
        """
        Organize the config to facillate .log call. Populate self.run_args

        Note:

        self.run_args = {
            tag: {
                func: {
                    freq: [
                        ([loggers], [capture]),
                        ([loggers2], [capture2]),
                        ...
                    ]
                },
                func2: {...}
            },
            tag2: {...}
        }

        """
        for tag, func_args in self.config.items():
            for func_arg in func_args:
                func = func_arg.get("func")

                leaf_loggers = []
                for logger_id in func_arg.get("uses", []):
                    leaf_loggers.append(self.leaf_logger[logger_id])

                self.run_args[tag][func][func_arg.get("freq", 1)].append(
                    (leaf_loggers, func_arg.get("capture"))
                )

    def log(
        self,
        value: Any,
        log_type: str,
        tag: str,
        capture: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Send args to the leaf loggers if the given tag, func, freq are accpeted. Need to
         pass three filters to be accepted.

         1. The provided tag must be a subset or regex match with the tags in the
            root logger config file
         2. The number of calls to the current self.log(...) must be a multiple of
            freq from the config file wrt tag and func
         3. If capture is speficied in the config file (only for metric log), it must be
            a subset or have a regex match

        If accepted, value=func(value) if func is provided, and pass this value to the
        leaf loggers

        :param value: Any value to log, may be dimentionally reduced by func
        :param log_type: String representing the root logger level
        :param tag: Candidate id that will be used to filter out only the wanted log
        :param capture: The property or dict key to record if match exists. If set to None
            no capture filter will be applied even if set in config

        """
        for tag_from_config, tag_run_args in self.run_args.items():
            if is_match_found(tag_from_config, tag):

                # key: func_name, value: {freq: {...}}
                for func_from_config, func_execute_args in tag_run_args.items():

                    # key: freq, value = [ ([loggers], [capture]), ... ]
                    for freq_from_config, execute_args in func_execute_args.items():

                        # increment the counter
                        self.inc(tag_from_config, func_from_config)

                        # execute_arg = ([loggers], [capture])
                        for execute_arg in execute_args:

                            leaf_loggers, captures_from_config = execute_arg

                            # check if the given tag.func is a multiple of the counter
                            if self.should_execute_on_frequency(
                                tag_from_config, func_from_config, freq_from_config
                            ):
                                # Capture filter (filter by class prop or dict key)
                                if capture is None:
                                    self._apply_func_and_log(
                                        value=value,
                                        tag=tag,
                                        log_type=log_type,
                                        func_from_config=func_from_config,
                                        leaf_loggers=leaf_loggers,
                                        *args,
                                        **kwargs,
                                    )

                                else:
                                    for capture_from_config in captures_from_config:
                                        if is_match_found(capture_from_config, capture):
                                            for (
                                                captured,
                                                value,
                                            ) in unravel_value_as_generator(value):

                                                self._apply_func_and_log(
                                                    value=value,
                                                    tag=tag,
                                                    log_type=log_type,
                                                    func_from_config=func_from_config,
                                                    leaf_loggers=leaf_loggers,
                                                    capture=captured,
                                                    *args,
                                                    **kwargs,
                                                )

    def _apply_func_and_log(
        self,
        value: Any,
        log_type: str,
        tag: str,
        func_from_config: str,
        leaf_loggers: list,
        *args,
        **kwargs,
    ):
        if func_from_config is not None:
            func_callable = import_from_registry(func_from_config)
            value = func_callable(value)

        for leaf_logger in leaf_loggers:
            leaf_logger.log(
                value=value,
                tag=tag,
                func=func_from_config,
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
        super().__init__(config, leaf_logger)

    def log(self, *args, **kwargs):
        super().log(log_type=self.LOG_TYPE, *args, **kwargs)
