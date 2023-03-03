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
Helper functions for generating metric function configs
"""

__all__ = ["data_logging_config_from_predefined"]

import logging
import os
import textwrap
from typing import Any, Dict, List, Optional, Union

import yaml

from deepsparse.loggers.build_logger import parse_out_predefined_function_groups
from deepsparse.loggers.config import MetricFunctionConfig
from deepsparse.loggers.metric_functions.registry import DATA_LOGGING_REGISTRY


_WHITESPACE = "  "
_LOGGER = logging.getLogger(__name__)


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

    if loggers is None:
        loggers = {"python": {}}

    metric_functions = [
        MetricFunctionConfig(func=group_name, frequency=frequency)
        for group_name in group_names
    ]
    data_logging_config = parse_out_predefined_function_groups(
        metric_functions=metric_functions, registry=registry
    )
    data_logging_config_str = _data_logging_config_string(data_logging_config)
    loggers_config_str = _loggers_to_config_string(loggers)

    config_str = loggers_config_str + "\n\n" + data_logging_config_str

    if save_dir:
        # save and log
        save_path = os.path.join(save_dir, save_name)
        parsed_data = yaml.safe_load(config_str)
        with open(save_path, "w") as file:
            yaml.dump(
                parsed_data,
                file,
                default_flow_style=False,
                line_break="\n",
                sort_keys=False,
            )
        _LOGGER.info(f"Saved data logging config to {save_path}")

    return config_str


def _loggers_to_config_string(
    loggers: Dict[str, Optional[Union[str, List[str]]]]
) -> str:
    lines = [_WHITESPACE + line for line in _nested_dict_to_lines(loggers)]
    lines.insert(0, "loggers:")
    return ("\n").join(lines)


def _data_logging_config_string(
    data_logging_config: Dict[str, List[MetricFunctionConfig]],
) -> str:
    lines = [_WHITESPACE + line for line in _nested_dict_to_lines(data_logging_config)]
    lines.insert(0, "data_logging:")
    return ("\n").join(lines)


def _nested_dict_to_lines(
    value: Any,
    key: Optional[str] = None,
    yaml_str_lines: Optional[List[str]] = None,
    _level: int = 0,
) -> List[str]:
    # converts a nested dictionary to a list of yaml string lines
    if yaml_str_lines is None:
        yaml_str_lines = []

    identation = _WHITESPACE * _level

    for new_key, new_value in value.items():
        if isinstance(new_value, dict):
            yaml_str_lines.append(f"{identation}{new_key}:")
            yaml_str_lines = _nested_dict_to_lines(
                new_value, new_key, yaml_str_lines, _level + 1
            )
        elif isinstance(new_value, list):
            list_as_str = _metric_functions_configs_to_string(new_value)
            yaml_str_lines.append(
                f"{new_key}:\n{textwrap.indent(list_as_str, prefix=_WHITESPACE)}"
            )
        else:
            yaml_str_lines.append(f"{identation}{new_key}: {new_value}")

    return yaml_str_lines


def _str_list_to_yaml(list_to_convert: List[str]) -> str:
    # converts a list of strings to their appropriate yaml string representation
    lines_indented = [
        textwrap.indent(line, prefix=_WHITESPACE) for line in list_to_convert
    ]
    lines_leading_coma = ["-" + line[1:] for line in lines_indented]
    return ("\n").join(lines_leading_coma)


def _metric_functions_configs_to_string(
    metric_functions_configs: List[MetricFunctionConfig],
) -> str:
    # converts a list of metric function configs to
    # their appropriate yaml string representation
    return _str_list_to_yaml(
        [
            _metric_function_config_to_string(config)
            for config in metric_functions_configs
        ]
    )


def _metric_function_config_to_string(
    metric_function_config: MetricFunctionConfig,
) -> str:
    # converts a single metric function config to
    # its appropriate yaml string representation
    text = (
        f"func: {metric_function_config.func}\n"
        f"frequency: {metric_function_config.frequency}"
    )

    target_loggers = metric_function_config.target_loggers
    # if target_loggers is not None,
    # include it in the yaml string
    if target_loggers:
        text += f"\ntarget_loggers:\n{textwrap.indent(_str_list_to_yaml(target_loggers), prefix=_WHITESPACE)}"  # noqa E501
    return text
