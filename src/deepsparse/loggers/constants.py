"""
Holds logging-related objects with constant values
"""
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

from dataclasses import dataclass
from enum import Enum


__all__ = [
    "MetricCategories",
    "validate_identifier",
    "SystemGroups",
    "FROM_PREDEFINED",
]

UNSUPPORTED_IDENTIFIER_CHARS = {".", "[", "]"}
FROM_PREDEFINED = "predefined"


class MetricCategories(Enum):
    """
    Metric Taxonomy [for reference]
        CATEGORY - category of metric (System/Data)
            GROUP - logical group of metrics
                METRIC - individual metric
    """

    # Categories
    SYSTEM = "system"
    DATA = "data"


@dataclass(frozen=True)
class SystemGroups:
    # Pipeline System Groups
    INFERENCE_DETAILS: str = "inference_details"
    PREDICTION_LATENCY: str = "prediction_latency"
    # Server System Groups
    REQUEST_DETAILS: str = "request_details"
    RESOURCE_UTILIZATION: str = "resource_utilization"


def validate_identifier(identifier: str):
    """
    Makes sure that the identifier does not contain any
    of the characters that would introduce ambiguity
    when parsing the identifier

    :param identifier: a string that is used
        to identify a log
    """
    for char in UNSUPPORTED_IDENTIFIER_CHARS:
        if char in identifier:
            raise ValueError(
                f"Logging identifier: {identifier} "
                f"contains unsupported character {char}"
            )
