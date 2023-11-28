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


from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import OperatorScheduler
from tests.deepsparse.v2.middleware.timer_middleware.utils import (
    TimedInferenceStateMiddlewarePipeline,
    TimedInferenceStatePipeline,
    TimedMiddlewarePipeline,
)
from tests.deepsparse.v2.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)


def test_pipeline_multiple_runtime_recoded_to_middleware_state():
    """Save recordings in the pipeline level into the middleware state"""
    AddThreePipeline = TimedMiddlewarePipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    timer = AddThreePipeline.timer_middleware.timer

    # check measurements added from TimedMiddlewarePipeline.run() exists
    assert "pipeline_state_time1" in timer.measurements
    assert "pipeline_state_time2" in timer.measurements


def test_inference_state_multiple_runtime_recoded_to_inference_state():
    """Save recordings in the inference level into the inference state"""
    AddThreePipeline = TimedInferenceStatePipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    timer = AddThreePipeline.timer_middleware.timer

    # inference state only saved to its state, not to the middleware state
    assert "op_state_time1" not in timer.measurements
    assert "op_state_time2" not in timer.measurements


# flake8: noqa
def test_both_pipeline_and_inference_state_multiple_runtime_recoded_to_middleware_state():
    """
    Save recordings in both the middleware state and inference level to the middleware state
    """
    AddThreePipeline = TimedInferenceStateMiddlewarePipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    timer = AddThreePipeline.timer_middleware.timer

    # inference state saved to its state and in middleware state
    assert "pipeline_state_time1" in timer.measurements
    assert "pipeline_state_time2" in timer.measurements
    assert "op_state_measurements" in timer.measurements
