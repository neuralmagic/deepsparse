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
Implementation of the Function Logger
"""
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.configs import PipelineLoggingConfig, TargetLoggingConfig
from deepsparse.loggers.metric_functions import apply_function


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw log values
    according to the specified config file

    :param logger: A DeepSparse Logger object
    :param config: A configuration dictionary that specifies the mapping between
        the target name and the functions to be applied to raw log values

        `config` can be specified as a dictionary type:
        e.g.

        {"pipeline_inputs":
            [{"function": "some_function_1",
              "frequency": 3},

            [{"function": "some_function_2",
              "frequency": 5}],

         "pipeline_outputs":
            [{"function": "some_function_1",
            "frequency": 4}]
        }

        or a a yaml file:
        e.g.

        ```yaml
        pipeline_inputs:
            - function: some_function_1
              frequency: 3
            - function: some_function_2
              frequency: 5
        pipeline_outputs:
            - function: some_function_1
              frequency: 4
        ```
    """

    def __init__(
        self,
        logger: BaseLogger,
        config: Union[List[PipelineLoggingConfig], List[TargetLoggingConfig]],
    ):

        self.logger = logger
        self.config = config
        # self.config = (
        #     config
        #     if isinstance(config, dict)
        #     else yaml.safe_load(Path(config).read_text())
        # )

        self.function_call_counter = self._create_frequency_counter(config)

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        if category == MetricCategories.DATA:
            # logging data information
            pipeline_name, *target = identifier.split(".")
            self._log_data(
                pipeline_name=pipeline_name,
                target=".".join(target),
                value=value,
                category=category,
            )
        else:
            # logging system information
            self.logger.log(
                identifier=identifier,
                value=value,
                category=category,
            )

    def _log_data(
        self,
        pipeline_name: str,
        target: str,
        value: Any,
        category: MetricCategories,
    ):
        list_target_functions = self.config.get(target)
        if list_target_functions is None:
            # no function to apply to the given target
            return

        for function_dict in list_target_functions:
            function = function_dict.get("function")
            logging_frequency = function_dict.get("frequency")

            if self.function_call_counter[target][function] % logging_frequency == 0:
                # reset the counter
                self.function_call_counter[target][function] = 0

                # fetch the function and apply it to the value
                mapped_value = apply_function(value=value, function=function)

                self.logger.log(
                    identifier=f"{pipeline_name}.{target}.{function}",
                    value=mapped_value,
                    category=category,
                )

            # increment the counter
            self.function_call_counter[target][function] += 1

    @staticmethod
    def _create_frequency_counter(config: List[PipelineLoggingConfig]):
        function_call_counter = defaultdict(
            lambda: defaultdict(lambda: defaultdict(str))
        )
        for pipeline_logging_config in config:
            targets = pipeline_logging_config.targets
            for target in targets:
                for mapping in target.mappings:
                    function_call_counter[pipeline_logging_config.name][target.target][
                        mapping.function_name
                    ] = 0

        return function_call_counter
