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
from typing import Any, Dict, List, Union

import yaml

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.metric_functions import apply_function


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw log values
    according to the specified config file

    :param loggers: A DeepSparse Logger object
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
        config: Union[str, Dict[str, List[Dict[str, Any]]]],
    ):

        self.logger = logger
        self.config = (
            config
            if isinstance(config, dict)
            else yaml.safe_load(Path(config).read_text())
        )
        self.function_call_counter = self._create_function_call_counter(self.config)

    def log(
        self, pipeline_name: str, target: str, value: Any, category: MetricCategories
    ):
        """
        :param pipeline_name: The name of the pipeline that the log relates to
        :param target: The identifier of the log
            The target may be a single string:
                e.g. target = "pipeline_input"
            Or, if identifier is complex, multiple strings are to be concatenated
            using dot as a separator:
                e.g. target = "pipeline_input.embedding"
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        if category == MetricCategories.DATA:
            # logging data information
            self._log_data(
                pipeline_name=pipeline_name,
                target=target,
                value=value,
                category=category,
            )
        else:
            # logging system information
            self.logger.log(
                pipeline_name=pipeline_name,
                target=target,
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
                    pipeline_name=pipeline_name,
                    target=target,
                    value=mapped_value,
                    category=category,
                )

            # increment the counter
            self.function_call_counter[target][function] += 1

    @staticmethod
    def _create_function_call_counter(config):
        function_call_counter = defaultdict(lambda: defaultdict(str))
        for target, list_functions in config.items():
            for function_dict in list_functions:
                function = function_dict.get("function")
                function_call_counter[target][function] = 0
        return function_call_counter
