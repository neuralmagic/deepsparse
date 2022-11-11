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
import re
from typing import Any, Optional, Sequence, Tuple, Union


__all__ = ["match_and_extract", "do_slicing_and_indexing"]


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

    for sub_remainder in remainder.split("."):
        # check whether sub_remainder contains square brackets
        # and thus needs slicing/indexing e.g. `some_key[0:2, 0]`
        square_brackets, sub_remainder = _check_square_brackets(sub_remainder)
        if not hasattr(value, sub_remainder):
            raise ValueError(
                "Attempting to access an non existing "
                f"attribute {sub_remainder} of an object {value}"
            )
        value = getattr(value, sub_remainder)

        if square_brackets:
            value = do_slicing_and_indexing(
                value=value, square_brackets=square_brackets
            )

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


def do_slicing_and_indexing(
    value: Union[Sequence, "numpy.ndarray", "torch.Tensor"],  # noqa: F821
    square_brackets: str,
) -> Any:
    """
    Perform slicing and/or indexing on the provided value

    Supported operations:
    - indexing: e.g value[0] or value[-2]
    - slicing: e.g value[0:2] or value[1:-3]
    - a composition of both: e.g value[0:2, 0] or value[1:-3, -1]

    :param value: A sequential/tensor/array type variable to be indexed and/or sliced
    :param square_brackets: The string that contains the indexing and/or slicing
        information inside square brackets
    :return: The value of interest
    """
    for string_operator in square_brackets.split(","):
        if ":" in string_operator:
            # slicing
            operator = slice(*map(int, re.findall(r"-?\d+", string_operator)))
        else:
            # indexing
            operator = int(string_operator)

        value = value.__getitem__(operator)
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
