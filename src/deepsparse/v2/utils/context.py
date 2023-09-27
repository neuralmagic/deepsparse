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


from typing import Callable, List, NamedTuple

from deepsparse.v2.utils.types import OperatorSchema


__all__ = ["Context"]


class StageInfo(NamedTuple):
    operator: Callable
    input: OperatorSchema
    output: OperatorSchema


class Context:
    """
    Context contains the full history of operators and their inputs and outputs
    in a pipeline
    """

    def __init__(self):
        self.stages_executed: List[StageInfo] = []

    def update(self, operator: Callable, input: OperatorSchema, output: OperatorSchema):
        self.stages_executed.append(
            StageInfo(operator=operator, input=input, output=output)
        )
