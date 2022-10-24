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

from typing import Any, Dict, List, Union

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.metric_functions import apply_function


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw log values
    according to the specified config file

    :param loggers: A single (or list of) DeepSparse Logger object(s)
    :param config: A configuration dictionary that specifies the mapping between
        the target name and the functions to be applied to raw log values
        e.g.
        ```
        {"pipeline_inputs":
            [{"function": "identity",
              "target_logger": "python", # if not specified log to all available loggers
              "frequency": 3},

            [{"function": "identity",
              "frequency": 5}],

         "pipeline_outputs":
            [{"function": "identity",
            "frequency": 4}]
        }
    """

    def __init__(
        self,
        loggers: Union[BaseLogger, List[BaseLogger]],
        config: Dict[str, List[Dict[str, Any]]],
    ):

        self.loggers = loggers if isinstance(loggers, list) else [loggers]
        self.config = self._add_frequency_counter(config)

    @property
    def identifier(self) -> str:
        return "function"

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        :param identifier: The identifier of the log
             By default should consist of at least one string (name of the pipeline):
                e.g. identifier = "pipeline_name"
            If identifier is to be more complex, optional strings are to be concatenated
            using dot as a separator:
                e.g. identifier = "pipeline_name.some_argument_1.some_argument_2"

            Note: by convention, "some_argument_1" very often is
                  referred to as `target_name`
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        if category == MetricCategories.DATA:
            pipeline_name, target_name, *_ = identifier.split(".")
            list_functions = self.config.get(target_name)
            if list_functions is None:
                # NOTE: explicitly filtering values not present in the config
                return
           # TODO & unindent everything below
                for (
                    function_dict
                ) in list_functions:  # iterate over all available functions
                    if (
                        function_dict["frequency_counter"] % function_dict["frequency"]
                        == 0
                    ):
                        # if count matches the frequency, log
                        # otherwise, pass
                        self._log_data(identifier, value, category, function_dict)
                    # increment the counter
                    function_dict["frequency_counter"] += 1

        else:
            [
                logger.log(identifier=identifier, value=value, category=category)
                for logger in self.loggers
            ]

    def _log_data(
        self,
        identifier: str,
        value: Any,
        category: MetricCategories,
        function_dict: Dict[str, Any],
    ):
        # fetch the string signature & apply function
        # specified by the signature to the value
        function_signature = function_dict.get("function")
        value = apply_function(value=value, function_signature=function_signature)

        # filter the loggers if required by the config
        target_loggers = function_dict.get("target_logger")
        if target_loggers:
            valid_loggers = [
                logger for logger in self.loggers if logger.identifier in target_loggers
            ]
        else:
            valid_loggers = self.loggers

        [
            logger.log(
                identifier=".".join([identifier, function_signature]),
                value=value,
                category=category,
            )
            for logger in valid_loggers
        ]

    @staticmethod
    def _add_frequency_counter(config):
        for identifier, list_functions in config.items():
            for function_dict in list_functions:
                function_dict.update(frequency_counter=0)
        return config
