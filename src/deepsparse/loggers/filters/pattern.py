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

import re
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy


SCALAR_TYPES_TUPLE = (
    int,
    float,
    bool,
    str,
)


def is_match_found(
    pattern: str,
    string: Optional[str] = None,
):
    """
    Check if a given pattern matches a string.

    - If the `pattern` starts with "re:", it treats the `pattern` as a regular
      expression and searches for a match within the `string`.

    :param pattern: (str): The pattern to match, which can be a simple string or a
        regular expression (if it starts with "re:").
    :param string: (str, optional): The string to test against the pattern.
        Defaults to None.
    :return: bool: True if a match is found, False otherwise.

    Examples:
        >>> is_match_found("apple", "apple")
        True

        >>> is_match_found("apple", "apple pie")
        False

        >>> is_match_found("cherry", "apple pie")
        False

        >>> is_match_found(r"re:\d{3}-\d{2}-\d{4}", "123-45-6789") # noqa
        True

        >>> is_match_found(r"re:\d{3}-\d{2}-\d{4}", "abc-def-ghij") # noqa
        False
    """
    if string is not None:
        if pattern.startswith("re:"):
            comp = re.compile(pattern[3:])
            if comp.search(string) is not None:
                return True
        else:
            if pattern == string:
                return True
    return False


def unravel_value_as_generator(
    value: Any, capture: str = ""
) -> Generator[Tuple[str, Any], None, None]:
    """
    Recursively unravel a nested data structure and yield tuples of capture paths
    and corresponding values.

    :param value: The input value to be unraveled.
    :param capture: A string representing the current capture path.
        Defaults to an empty string.

    Yields:
        Generator[Tuple[str, Any], None, None]: A generator that yields tuples
        containing a capture path (string) and the corresponding value.

    Examples:
        >>> data = {'a': [1, 2, {'b': 3}], 'c': 4}
        >>> for path, val in unravel_value_as_generator(data):
        ...     print(f"Capture Path: {path}, Value: {val}")
        Capture Path: ['a'], Value: [1, 2, {'b': 3}]
        Capture Path: ['a'][0], Value: 1
        Capture Path: ['a'][1], Value: 2
        Capture Path: ['a'][2], Value: {'b': 3}
        Capture Path: ['a'][2]['b'], Value: 3
        Capture Path: ['c'], Value: 4
    """

    if isinstance(value, Dict):
        for key, val in value.items():
            new_capture = capture + f"['{key}']"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, numpy.ndarray):
        yield (capture, value)

    elif isinstance(value, Tuple) and not isinstance(value, SCALAR_TYPES_TUPLE):
        for idx, val in enumerate(value):
            new_capture = capture + f"[{idx}]"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, List):
        for idx, val in enumerate(value):
            new_capture = capture + f"[{idx}]"
            yield from unravel_value_as_generator(val, new_capture)

    elif isinstance(value, Enum):
        yield (capture.lstrip("."), value.value)

    elif isinstance(value, object) and not isinstance(value, SCALAR_TYPES_TUPLE):

        if hasattr(value, "__dict__"):
            for prop, val in vars(value).items():
                new_capture = capture + f".{prop}"
                yield from unravel_value_as_generator(val, new_capture)

        else:  # None type only
            yield (capture, None)

    else:
        # scalars: (int, float, bool, str)
        yield (capture, value)
