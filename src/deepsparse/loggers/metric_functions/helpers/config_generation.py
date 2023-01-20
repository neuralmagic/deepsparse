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

__all__ = ["data_logging_config_from_predefined"]

from typing import Any, Dict, List, Optional, Union

from deepsparse.loggers.build_logger import predefined_metric_function
from deepsparse.loggers.config import MetricFunctionConfig
from deepsparse.loggers.metric_functions.registry import DATA_LOGGING_REGISTRY


def data_logging_config_from_predefined(
    group_names: Union[str, List[str]],
    frequency: int = 1,
    loggers: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    save_dir: Optional[str] = None,
    save_name: str = "data_logging_config.yaml",
    registry: Dict[str, Any] = DATA_LOGGING_REGISTRY,
) -> str:
    """
    Generate a data logging config yaml string using a predefined
    function groups configuration.

    :param group_names: A single group name or a list of group names,
        that are to be translated into the yaml configuration.
    :param loggers: Defines set of loggers that will be used to collect
        the data logs. It is dictionary that maps the logger integration
        names to their initialization arguments
    :param frequency: Optional frequency of the data logging
        functions in the resulting yaml configuration. By default,
        set to 1
    :param save_dir: If provided, the resulting yaml configuration is
        saved to the provided directory
    :param save_name: If config is saved, it will be under this
        filename
    :return: A string yaml dict that specifies the data logging
        configuration
    """
    if isinstance(group_names, str):
        group_names = [group_names]

    metric_functions = [
        MetricFunctionConfig(func=group_name, frequency=frequency)
        for group_name in group_names
    ]
    data_logging_config = predefined_metric_function(
        metric_functions=metric_functions, registry=registry
    )
    return data_logging_config
