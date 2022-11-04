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
from typing import Any, Optional, Tuple


"""
Helpers functions for logging
"""

__all__ = ["match_and_extract"]


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
