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
from typing import Any, List

from pydantic import BaseModel, Field

from deepsparse.v2.utils import InferenceState


__all__ = ["SubGraph", "StreamingOutput"]


@dataclass
class SubGraph:
    """
    Helper dataclass to store information about each running sub graph.
    """

    step: int
    inf: InferenceState
    end: List[str]
    output: Any = None

    def parse_output(self, operator_output: Any):
        if isinstance(operator_output, tuple):
            operator_output, state_update = operator_output[0], operator_output[-1]
            self.inf.update_state(state_update)
        return operator_output


class StreamingOutput(BaseModel):
    """
    Helper object to store the output of a streaming operator. Facilitates
    returning data to be used in the next step of the pipeline and yielding
    the data immediately from the pipeline.
    """

    data_to_return: Any = Field(
        description="Any data that should be returned "
        "to be used in the next step of the pipeline"
    )
    data_to_yield: Any = Field(
        description="Any data that should be yielded to the user"
    )
