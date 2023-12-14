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
Simple example and testing middlewares in Ops and Pipeline
"""

from typing import Dict

from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.middlewares import MiddlewareManager, MiddlewareSpec
from deepsparse.operators import Operator
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import OperatorScheduler
from tests.deepsparse.middlewares import PrintingMiddleware, SendStateMiddleware


class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 2}


middlewares = [
    MiddlewareSpec(PrintingMiddleware),
    MiddlewareSpec(SendStateMiddleware),
]

middleware_manager = MiddlewareManager(middlewares)

AddThreePipeline = Pipeline(
    ops=[AddOneOperator(), AddTwoOperator()],
    router=LinearRouter(end_route=2),
    schedulers=[OperatorScheduler()],
    middleware_manager=middleware_manager,
)


def test_middleware_execution_in_pipeline_and_operator():
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    # SendStateMiddleware, order of calls:
    # Pipeline start, AddOneOperator start, AddOneOperator end
    # AddTwoOperator start, AddTwoOperator end, Pipeline_ end
    expected_order = [0, 0, 1, 0, 1, 1]
    state = AddThreePipeline.middleware_manager.state
    assert state["SendStateMiddleware"] == expected_order
