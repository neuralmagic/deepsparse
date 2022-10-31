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
from deepsparse.loggers.config import (
    MetricFunctionConfig,
    MultiplePipelinesLoggingConfig,
    PipelineLoggingConfig,
    TargetLoggingConfig,
)


__all__ = ["FunctionLogger"]


class FunctionLogger(BaseLogger):
    """
    DeepSparse logger that applies functions to raw log values
    according to the specified config file

    :param logger: A DeepSparse Logger object
    :param config: A data structure that specifies the configuration
        of the Logger. Can be one of the following:

        1. MultiplePipelinesLoggingConfig
        2. PipelineLoggingConfig
        3. TargetLoggingConfig
        4. A dictionary, e.g.
            {"target": "pipeline_inputs",
             "mappings": [{"func": "builtins:some_func_1",
                          "frequency": 3
                          },
                          {"func": ".../custom_functions.py:some_func_2",
                          frequency": 5,
                          }]
             }

            Note: To specify multiple targets, use PipelineLoggingConfig model

        5. String path to the yaml file, e.g.
            ```yaml
            targets:
            - target: pipeline_inputs
              mappings:
               - func: builtins:some_func_1
                 frequency: 3
               - func: .../custom_functions.py:some_func_2
                 frequency: 5
            - target: pipeline_outputs
              mappings:
               - func: builtins:some_func_3
                 frequency: 4
            ```
    """

    def __init__(
        self,
        logger: BaseLogger,
        config: Union[
            MultiplePipelinesLoggingConfig,
            PipelineLoggingConfig,
            TargetLoggingConfig,
            Dict[str, Dict[str, Any]],
            str,
        ],
    ):

        self.logger = logger
        self.config = self._parse_config(config)
        self.frequency_counter = defaultdict(
            lambda: (defaultdict(lambda: defaultdict(int)))
        )

    def log(self, identifier: str, value: Any, category: MetricCategories):
        """
        :param identifier: The name of the thing that is being logged.
        :param value: The data structure that the logger is logging
        :param category: The metric category that the log belongs to
        """
        if category == MetricCategories.DATA:
            # logging data information
            pipeline_name, *target = identifier.split(".")
            target = ".".join(target)
            self._log_data(
                pipeline_name=pipeline_name,
                target=target,
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
        metric_functions = self._match(pipeline_name=pipeline_name, target=target)
        if metric_functions is None:
            return

        for metric_function in metric_functions:
            function = metric_function.func
            function_name = metric_function.function_name
            logging_frequency = metric_function.frequency

            counts = self._get_frequency_counts(
                pipeline_name=pipeline_name, target=target, function_name=function_name
            )

            if counts % logging_frequency == 0:

                mapped_value = function(value)

                self.logger.log(
                    identifier=f"{pipeline_name}.{target}.{function_name}",
                    value=mapped_value,
                    category=category,
                )

                # reset the counter
                self.frequency_counter[pipeline_name][target][function_name] = 0

            # increment the counter
            self.frequency_counter[pipeline_name][target][function_name] += 1

    def _match(
        self, pipeline_name: str, target: str
    ) -> Optional[List[MetricFunctionConfig]]:
        # given the pipeline_name and target, try (if possible) to match them
        # with the appropriate list of metric functions
        for pipeline_logging_config in self.config.pipelines:
            expected_pipeline_name = pipeline_logging_config.name
            if (
                # if `expected_pipeline_name` specified
                # and does not match with the `pipeline_name`
                # skip to the next PipelineLoggingConfig
                expected_pipeline_name is not None
                and pipeline_name != expected_pipeline_name
            ):
                continue
            for target_logging_config in pipeline_logging_config.targets:
                expected_target = target_logging_config.target
                # if `expected_target` does not match
                # with the `target`
                # skip to another TargetLoggingConfig
                if expected_target != target:
                    continue
                else:
                    return target_logging_config.mappings
        # no matches found
        return None

    def _get_frequency_counts(
        self, pipeline_name: str, target: str, function_name: str
    ) -> int:
        # get the number of times the function was called in the context of
        # the pipeline and target
        counts = (
            self.frequency_counter.get(pipeline_name, {})
            .get(target, {})
            .get(function_name)
        )
        if not counts:
            counts = 0
            self.frequency_counter[pipeline_name][target][function_name] = counts
        return counts

    @staticmethod
    def _parse_config(
        config: Union[
            MultiplePipelinesLoggingConfig,
            PipelineLoggingConfig,
            TargetLoggingConfig,
            Dict[str, Dict[str, Any]],
            str,
        ]
    ) -> MultiplePipelinesLoggingConfig:
        # covert all the possible config inputs
        # to MultiplePipelinesLoggingConfig representation
        if isinstance(config, dict):
            config = TargetLoggingConfig(**config)
        if isinstance(config, str):
            config = PipelineLoggingConfig(**yaml.safe_load(Path(config).read_text()))
        if isinstance(config, TargetLoggingConfig):
            config = PipelineLoggingConfig(targets=[config])
        if isinstance(config, PipelineLoggingConfig):
            config = MultiplePipelinesLoggingConfig(pipelines=[config])
        return config
