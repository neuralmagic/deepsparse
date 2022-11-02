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

from typing import Any, Optional, Tuple

from pydantic import BaseModel


"""
Helpers functions for logging
"""

__all__ = ["possibly_extract_value", "match"]


def possibly_extract_value(
    value: Any, identifier_reminder: Optional[str] = None
) -> Any:
    """
    Given a string of ("."-separated) sub_identifiers, access the items
    inside `value`.

    :param value: A data structure that may potentially hold "nested" values
        of interest.
    :param sub_identifiers: A string of "."-separated keys that are used to
        access "nested" value of interest inside`value`.
    :return: Value of interest
    """
    if not identifier_reminder:
        return value

    value = dict(value) if isinstance(value, BaseModel) else value

    for reminder in identifier_reminder.split("."):
        # TODO: Add the support for slicing
        # e.g. pipeline_inputs.images[0,0,0,0]
        value = value[reminder]
    return value


def match(template: str, identifier: str) -> Tuple[bool, Optional[str]]:
    """
    Match the template against the identifier

    :param template: A string in format:
        <pipeline_name>.<target_name>.<reminder>
        or
        <target_name>.<reminder>
    :param identifier: A string in format <pipeline_name>.<target_name>
    :return: A tuple that consists of:
        - a boolean (True if match, False otherwise)
        - an optional reminder (string if matched, None otherwise)
    """
    pipeline, target = identifier.split(".")

    # assume template is "<target_name>.<...>"
    t_target, *t_other = template.split(".")
    if _match_two_strings(t_target, target):
        return True, ".".join(t_other) if t_other else None
    # if template is just <target_name>, no match made
    if not t_other:
        return False, None
    # assume template is "<pipeline_name>.<target_name>.<..>"
    else:
        # assume template is "<pipeline_name>.<target_name>.<..>"
        t_pipeline, t_target, *t_other = template.split(".")
        if _match_two_strings(t_pipeline, pipeline):
            if _match_two_strings(t_target, target):
                return True, ".".join(t_other) if t_other else None

    return False, None


def _match_two_strings(goal_string: str, string: str) -> bool:
    # TODO: possible regex logic will end up here
    if goal_string == string:
        return True
    return False
