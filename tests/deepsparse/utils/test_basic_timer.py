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

import time
from collections import defaultdict
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
        inference_state = kwargs.get("inference_state")
        with inference_state.time(id="AddOneOperator"):
            time.sleep(0.2)
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        inference_state = kwargs.get("inference_state")
        with inference_state.time(id="AddTwoOperator"):
            time.sleep(0.5)
        return {"value": inp.value + 2}


def test_pipeline_fine_grained_timer_record_operator_run_times():
    AddThreePipeline = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    measurements: defaultdict[list] = AddThreePipeline.timer_manager.measurements[0]

    assert len(measurements) == 2
    expected_keys = {"AddTwoOperator", "AddOneOperator"}
    for key in measurements.keys():
        expected_keys.remove(key)
    assert len(expected_keys) == 0


def test_pipelines_with_shared_timer_manager():
    """
    Share the timer_manager, check that entries are
    in the expected format
    """
    AddThreePipeline = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8
    assert len(AddThreePipeline.timer_manager.measurements) == 1

    shared_timer_manager = AddThreePipeline.timer_manager
    AddThreePipeline2 = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        timer_manager=shared_timer_manager,
    )
    pipeline_output2 = AddThreePipeline2(pipeline_input)

    assert pipeline_output2.value == 8

    measurements = AddThreePipeline.timer_manager.measurements
    assert len(measurements) == 2

    assert (
        AddThreePipeline.timer_manager.measurements
        == AddThreePipeline2.timer_manager.measurements
    )

    pipeline1_measuremnts = measurements[0]
    pipeline2_measuremnts = measurements[1]

    # Check that the keys are the same, and running two identical pipeline runtimes
    # are reproducible within delta
    delta = 0.001
    for key in pipeline1_measuremnts.keys():
        assert key in pipeline2_measuremnts
        print(abs(pipeline1_measuremnts[key][0] - pipeline2_measuremnts[key][0]))
        assert delta > abs(
            pipeline1_measuremnts[key][0] - pipeline2_measuremnts[key][0]
        )
