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
import warnings
from difflib import SequenceMatcher
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy

import deepsparse.loggers.metric_functions as built_ins
from deepsparse.loggers import MetricCategories


__all__ = [
    "match_and_extract",
    "get_function_and_function_name",
    "NO_MATCH",
    "access_nested_value",
    "finalize_identifier",
]

NO_MATCH = "NO_MATCH"


def finalize_identifier(
    identifier: str,
    category: MetricCategories,
    function_name: str,
    remainder: Optional[str] = None,
) -> str:
    """
    Compose the final identifier string from the identifier, category, function name

    :param identifier: The identifier string
    :param category: The category of the identifier
    :param function_name: The name of the function applied to the identifier
    :param remainder: The remainder of the identifier after the matching was applied
    :return: The final identifier string
    """
    if remainder:
        if category == MetricCategories.DATA:
            # if remainder has slicing/indexing/access information,
            # remove the square brackets:
            remainder = remainder.split("[")[0]
        # join the identifier and remainder
        identifier += "." + remainder

    if category == MetricCategories.DATA:
        # if the category is DATA, add the function name to the identifier
        identifier += f"__{function_name}"

    return identifier


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
        import torch

        function, function_name = _get_function_and_function_name_from_framework(
            framework=torch, function_identifier=function_identifier
        )
        return function, function_name

    if function_identifier.startswith("numpy.") or function_identifier.startswith(
        "np."
    ):
        function, function_name = _get_function_and_function_name_from_framework(
            framework=numpy, function_identifier=function_identifier
        )
        return function, function_name

    if len(function_identifier.split(":")) == 2:
        # assume a dynamic import function of the form
        # '<path_to_the_python_script>:<function_name>'
        path, func_name = function_identifier.split(":")
        if not os.path.exists(path):
            raise ValueError(f"Path to the metric function file {path} does not exist")
        spec = importlib.util.spec_from_file_location(
            "user_defined_metric_functions", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, func_name), func_name

    func_name = function_identifier
    if not hasattr(built_ins, func_name):
        raise ValueError(f"Metric function {func_name} not found")
    return getattr(built_ins, func_name), func_name


def match_and_extract(
    template: str,
    identifier: str,
    value: Any,
    category: MetricCategories,
) -> Tuple[Any, Optional[str]]:
    """
    Attempts to match the template against the identifier. If successful, and
    the category is DATA, uses the remainder to extract the item of interest inside
    `value` data structure.

    :param template: A string that defines the matching criteria
    :param identifier: A string that will be compared with the template, may
        be a full string to match to a logged target with access to properties
        and indices, a regex pattern to match to prefixed by `re:`, or a
        MetricCategory to match to prefixed by `category:`
    :param value: Raw value from the logger
    :param category: optional MetricCategory of the value
    :return: Tuple:
        - Value of interest or string flag that indicates that there was no match
        - An optional remainder string
    """
    is_match, remainder = check_identifier_match(template, identifier)

    if is_match:
        if category == MetricCategories.SYSTEM:
            return value, remainder
        else:
            return possibly_extract_value(value, remainder), remainder
    else:
        return NO_MATCH, remainder


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

    for sub_remainder in remainder.split("."):
        # check whether sub_remainder contains square brackets
        # and thus needs slicing/indexing e.g. `some_key[0:2, 0]`
        square_brackets, sub_remainder = _check_square_brackets(sub_remainder)
        if sub_remainder:
            if not hasattr(value, sub_remainder):
                raise ValueError(
                    "Attempting to access an non existing "
                    f"attribute {sub_remainder} of an object {value}"
                )
            value = getattr(value, sub_remainder)

        if square_brackets:
            value = access_nested_value(value=value, square_brackets=square_brackets)

    return value


def _check_square_brackets(sub_remainder: str) -> Tuple[str, str]:
    # split the sub_remainder into two parts:
    # 1. the sub_remainder string
    # 2. a list of consecutive indexing/slicing operations
    sub_remainder, *square_brackets = sub_remainder.split("[")
    # join the list of consecutive indexing/slicing operations
    square_brackets = ",".join(
        [re.search(r"(.*?)\]", x).group(1) for x in square_brackets]
    )
    if square_brackets:
        # "some_value[0,2:4]" -> ("0,2:4", "some_value")
        # "some_value[0][1:3]" -> ("0,1:3", "some_value")
        return square_brackets, sub_remainder
    else:
        # "some_value" -> ("", "some_value")
        return "", sub_remainder


def access_nested_value(
    value: Union[Sequence, numpy.ndarray, "torch.Tensor", Dict[str, Any]],  # noqa F821
    square_brackets: str,
) -> Any:
    """
    Use the contents of the `square_brackets` to access
    the item nested inside `value`.

    Supported operations:
    - indexing: e.g value[0] or value[-2]
    - slicing: e.g value[0:2] or value[1:-3]
    - dictionary access e.g. value["key"] or value["key1"]["key2"]
    - a composition of the above:
        e.g value[0:2][0]
            value[1:-3, -1]
            value["key1"][0]["key2"]
            value["key1"][0:2]["key2"]

    :param value: A sequential/tensor/array type variable to be indexed and/or sliced
    :param square_brackets: The string that contains the indexing and/or slicing,
        and/or dictionary access information
    :return: The value of interest
    """
    for string_operator in square_brackets.split(","):
        # check whether `string_operator` contains at least one character
        # -> must be a dictionary key
        if string_operator.upper().isupper():
            # dictionary access
            operator = string_operator.replace("'", "").replace(
                '"', ""
            )  # remove the quotes
            if not isinstance(value, dict):
                raise ValueError(
                    f"Attempting to access a dictionary key {operator} "
                    f"of a non-dictionary object {value}"
                )
        elif ":" in string_operator:
            # slicing
            operator = slice(*map(int, re.findall(r"-?\d+", string_operator)))
        else:
            # indexing
            operator = int(string_operator)

        _warn_if_array_or_tensor(value)
        value = value.__getitem__(operator)

    return value


def check_identifier_match(
    template: str,
    identifier: str,
) -> Tuple[bool, Optional[str]]:
    """
    Match the template against the identifier

    :param template: A string the in format:
        <string_n-t>/<string_n-t+1)>/<...>/<string_n>(optionally).<remainder>,
        a regex pattern prefixed by `re:`

    :param identifier: A string in the format:
        <string_n-t>/<string_n-t+1)>/<...>/<string_n>

    :return: A tuple that consists of:
        - a boolean (True if match, False otherwise)
        - an optional remainder (string if matched, None otherwise)
    """
    if template[:3] == "re:":
        pattern = template[3:]
        return re.match(pattern, identifier) is not None, None

    match = SequenceMatcher(None, identifier, template).find_longest_match(
        0, len(identifier), 0, len(template)
    )
    if not match:
        return False, None

    if match.b == 0:
        """
        The template and identifier share common components.
        There is a potential match.
        Case: 0) identifier = "foo/bar" and template: "foo/bar"
            results in match and remainder None
        Case: 1) identifier: "foo/bar" and template: "foo/bar.baz"
            results in match and remainder "baz"
        Case 2) identifier: "foo/bar/alice" and template: "bar"
            results in match and remainder None
        """
        if match.size == len(identifier) == len(template):
            # case 0)
            return True, None
        possible_remainder = (
            identifier[match.a + match.size :] or template[match.b + match.size :]
        )
        if possible_remainder.startswith(".") and match.a == match.b == 0:
            # case 1)
            return True, possible_remainder[1:]

        if possible_remainder.startswith("/"):
            # case 2
            return True, None

    return False, None


def _get_function_and_function_name_from_framework(
    framework: ModuleType, function_identifier: str
) -> Tuple[Callable[[Any], Any], str]:
    func_attributes = function_identifier.split(".")[1:]
    module = framework
    for attribute in func_attributes:
        module = getattr(module, attribute)
    return module, ".".join(func_attributes)


def _warn_if_array_or_tensor(value: Any) -> Any:
    msg = """
    If value is an array or tensor, one should refrain from 
    slicing/indexing/accessing its elements using 'access_nested_value'.
    This function should only be used for Sequence types 
    (lists, tuple, ranges, etc.) or Dictionaries due to their simple structure.
    For more complex operations on `value`, one should use `metric_functions`
    specified directly in the data logging config"""  # noqa W291

    if isinstance(value, numpy.ndarray) or hasattr(value, "numpy"):
        warnings.warn(msg)
