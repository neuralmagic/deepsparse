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
Pydantic Models for Logging Configs
"""

from typing import List, Optional

from pydantic import BaseModel, Field, validator


__all__ = [
    "MetricFunctionConfig",
]


class MetricFunctionConfig(BaseModel):
    """
    Holds logging configuration for a metric function
    """

    func: str = Field(
        description="The name that specifies the metric function to be applied. "
        "It can be: "
        "1) a built-in function name "
        "2) a dynamic import function of the form "
        "'<path_to_the_python_script>:<function_name>' "
        "3) a framework function (e.g. np.mean or torch.mean)"
    )

    frequency: int = Field(
        description="Specifies how often the function should be applied"
        "(measured in numbers of inference calls).",
        default=1,
    )

    target_loggers: Optional[List[str]] = Field(
        default=None,
        description="Overrides the global logger configuration in "
        "the context of the DeepSparse server. "
        "If not None, this configuration stops logging data "
        "to globally specified loggers, and will only use "
        "the subset of loggers (specified here by a list of their names).",
    )

    @validator("frequency")
    def non_zero_frequency(cls, frequency: int) -> int:
        if frequency <= 0:
            raise ValueError(
                f"Passed frequency: {frequency}, but "
                "frequency must be a positive integer greater equal 1"
            )
        return frequency
