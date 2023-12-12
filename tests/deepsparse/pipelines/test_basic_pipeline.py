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
Simple example and test of a dummy pipeline
"""

from typing import Dict

from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.operators import Operator
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import OperatorScheduler


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


# AddThreePipeline = Pipeline(
#     ops=[AddOneOperator(), AddTwoOperator()],
#     router=LinearRouter(end_route=2),
#     schedulers=[OperatorScheduler()],
# )

# AddThreePipeline2 = Pipeline(
#     ops=[AddOneOperator(), AddTwoOperator()],
#     router=LinearRouter(end_route=2),
#     schedulers=[OperatorScheduler()],
# )


def test_run_simple_pipeline():
    pipeline_input = IntSchema(value=5)
    AddThreePipeline = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )

    pipeline_output = AddThreePipeline(pipeline_input)
    breakpoint()
    
    AddThreePipeline2 = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    p2 = AddThreePipeline2(pipeline_input)
    
    

    assert pipeline_output.value == 8
    breakpoint()

"""


AddThreePipeline.container.timer_service().measurements()
AddThreePipeline2.container.timer_service().measurements()
"""