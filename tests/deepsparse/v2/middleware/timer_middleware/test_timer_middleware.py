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


def test_pipeline_multiple_runtime_successfully_recoded():
    AddThreePipeline = TimedMiddlewarePipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    timer = AddThreePipeline.timer_middleware.timer
    assert "foo" in timer.measurements
    assert "bar" in timer.measurements

    time_delta = 0.005
    assert abs(1 - timer.measurements["foo"]) < time_delta
    assert abs(0.5 - timer.measurements["bar"]) < time_delta


def test_inference_state_multiple_runtime_successfully_recoded():
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
    assert "foo" not in timer.measurements
    assert "bar" not in timer.measurements


def test_both_pipeline_and_inference_state_multiple_runtime_successfully_recoded():
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
    assert "foo" in timer.measurements
    assert "bar" in timer.measurements
    assert "inference_state" in timer.measurements
