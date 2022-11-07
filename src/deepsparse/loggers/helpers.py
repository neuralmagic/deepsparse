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
Helpers functions for logging
"""

import importlib
import os.path
import re
from typing import Any, Callable, Optional, Tuple

import numpy

import deepsparse.loggers.metric_functions.built_ins as built_ins
import torch


__all__ = ["match_and_extract", "get_function_and_function_name"]


def get_function_and_function_name(
    function_identifier: str,
) -> Tuple[Callable[[Any], Any], str]:
    """
    Parse function identifier and return the function as well as its name

    :param function_identifier: Can be one of the following:

        1. framework function, e.g.
            "torch.mean" or "numpy.max"

        2. built-in function, e.g.
            "function_name"
            note: function needs to be available in the module
            with built-in functions, see the imports above)

        3. user-defined function in the form of
           '<path_to_the_python_script>:<function_name>', e.g.
           "{...}/script_name.py:function_name"

    :return: A tuple (function, function name)
    """

    if function_identifier.startswith("torch."):
        func_name = function_identifier.split(".")[1]
        return getattr(torch, func_name), func_name

    if function_identifier.startswith("numpy.") or function_identifier.startswith(
        "np."
    ):
        func_name = function_identifier.split(".")[1]
        return getattr(numpy, func_name), func_name

    if len(function_identifier.split(":")) == 2:
        # assume a dynamic import function of the form
        # '<path_to_the_python_script>:<function_name>'
        path, func_name = function_identifier.split(":")
        if not os.path.exists(path):
            raise ValueError(f"Path to the python script {path} does not exist")
        spec = importlib.util.spec_from_file_location(
            "user_defined_metric_functions", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name), func_name

    func_name = function_identifier
    return getattr(built_ins, func_name), func_name


def match_and_extract(template: str, identifier: str, value: Any) -> Optional[Any]:
    """
    Attempts to match the template against the identifier. If successful,
    uses the remainder to extract the item of interest inside `value` data structure.

    :param template: A string that defines the matching criteria
    :param identifier: A string that will be compared with the template
    :param value: Raw value from the logger
    :return: (Optional) Value of interest
    """
    is_match, remainder = check_identifier_match(template, identifier)
    return possibly_extract_value(value, remainder) if is_match else None


def possibly_extract_value(value: Any, remainder: Optional[str] = None) -> Any:
    """
    Given a remainder (string of "."-separated strings), try to
    access the items inside `value` data structure.

    :param value: A data structure that may potentially hold "nested" values
        of interest.
    :param remainder: A string of "."-separated keys that are used to
        access "nested" value of interest inside`value`.
    :return: Value of interest
    """
    if not remainder:
        return value

    for sub_remainders in remainder.split("."):
        value = value.__getattribute__(sub_remainders)

    return value


def check_identifier_match(
    template: str, identifier: str
) -> Tuple[bool, Optional[str]]:
    """
    Match the template against the identifier

    :param template: A string the in format:
        <string_n-t>.<string_n-t+1)>.<...>.<string_n>(optionally).<remainder>
        or
        a regex pattern

    :param identifier: A string in the format:

        1.  <string_n-t>.<string_n-t+1)>.<...>.<string_n>
        2.
            if template and identifier do not share any first
            <string_n-t+k> components, there is no match

    :return: A tuple that consists of:
        - a boolean (True if match, False otherwise)
        - an optional remainder (string if matched, None otherwise)
    """
    if template == identifier:
        return True, None
    if template.startswith(identifier):
        return True, template.replace(identifier, "")[1:]
    if template[:3] == "re:":
        pattern = template[3:]
        return re.match(pattern, identifier) is not None, None

    return False, None
