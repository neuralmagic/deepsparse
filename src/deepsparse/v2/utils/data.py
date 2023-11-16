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
from typing import Any

from deepsparse.v2.utils import InferenceState


__all__ = ["SplitRoute"]


@dataclass
class SplitRoute:
    """
    Helper dataclass to store information about each running split route.
    """

    step: int
    inf: InferenceState
    output: Any = None

    def parse_output(self, operator_output: Any):
        if isinstance(operator_output, tuple):
            state_update = operator_output[-1]
            operator_output = operator_output[0]
            self.inf.update_state(state_update)
        return operator_output
