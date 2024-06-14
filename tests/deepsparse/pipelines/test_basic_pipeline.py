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

import asyncio
from typing import Dict

from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.operators import Operator
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import OperatorScheduler
from deepsparse.utils.state import InferenceState


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


AddThreePipeline = Pipeline(
    ops=[AddOneOperator(), AddTwoOperator()],
    router=LinearRouter(end_route=2),
    schedulers=[OperatorScheduler()],
)


def test_run_simple_pipeline():
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8


def test_run_async_simple_pipeline():
    test_actually_ran = False

    async def _actual_test():
        nonlocal test_actually_ran

        inference_state = InferenceState()
        inference_state.create_state({})
        pipeline_input = IntSchema(value=5)

        pipeline_output = await AddThreePipeline.run_async(
            pipeline_input, inference_state=inference_state
        )

        assert pipeline_output.value == 8

        test_actually_ran = True

    asyncio.run(_actual_test())
    assert test_actually_ran
