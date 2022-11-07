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
from typing import Any, List

from deepsparse.loggers import BaseLogger, MetricCategories
from deepsparse.loggers.helpers import match_and_extract


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw,
    logged values (collected in the log() method)
    according to FunctionLogger's attributes

    :param logger: A child DeepSparse Logger object
    :param target_logging_configs: A list of TargetLoggingConfig objects

    """

    def __init__(
        self,
        logger: BaseLogger,
        target_logging_configs: List["TargetLoggingConfig"], # noqa F821
    ):

        self.logger = logger
        self.target_logging_configs = target_logging_configs
        self._function_call_counter = defaultdict(lambda: defaultdict(int))

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        Collect information from the pipeline and:
        1) Filter it according to the `self.target_logging_configs`
        2) Apply metric functions according to the `self.target_logging_configs`

        :param identifier: The name of the item that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """

        for target_cfg in self.target_logging_configs:
            target_identifier = target_cfg.target
            target_mappings = target_cfg.mappings

            for metric_func_cfg in target_mappings:

                frequency = metric_func_cfg.frequency
                function = metric_func_cfg.func
                function_name = metric_func_cfg.function_name

                extracted_value = match_and_extract(
                    template=target_identifier, identifier=identifier, value=value
                )

                if extracted_value is not None:
                    if self._should_counter_log(
                        target_identifier, function_name, frequency
                    ):
                        mapped_value = function(extracted_value)
                        self.logger.log(
                            identifier=f"{identifier}.{function_name}",
                            value=mapped_value,
                            category=category,
                        )
                        self._reset_counter(target_identifier, function_name)
                    self._increment_counter(target_identifier, function_name)

    def _should_counter_log(
        self, target_identifier: str, function_name: str, frequency: int
    ) -> bool:
        count = self._function_call_counter.get(target_identifier, {}).get(
            function_name
        )
        if not count:
            count = 0
            self._function_call_counter[target_identifier][function_name] = count
        return count % frequency == 0

    def _reset_counter(self, target_identifier: str, function_name: str):
        self._function_call_counter[target_identifier][function_name] = 0

    def _increment_counter(self, target_identifier: str, function_name: str):
        self._function_call_counter[target_identifier][function_name] += 1
