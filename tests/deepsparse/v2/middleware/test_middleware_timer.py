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


# from tests.deepsparse.v2.middleware.utils import CounterMiddleware
from deepsparse.v2.middleware import TimerMiddleware
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from tests.deepsparse.v2.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)


def test_timer_saved_in_pipeline_and_inference():
    """Check that timer gets saved in Pipeline and Ops"""
    timer_middleware = TimerMiddleware()
    middleware = [timer_middleware]
    AddThreePipeline = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware=middleware,
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    assert bool(timer_middleware.start_time) is False
    assert bool(timer_middleware.measurements) is True

    expected_keys = {"Pipeline", "AddOneOperator", "AddTwoOperator"}
    for key, _ in timer_middleware.measurements.items():
        expected_keys.remove(key)
    assert len(expected_keys) == 0

    pipeline_time = timer_middleware.measurements["Pipeline"]
    assert timer_middleware.measurements["AddOneOperator"] < pipeline_time
    assert timer_middleware.measurements["AddTwoOperator"] < pipeline_time

    assert (
        timer_middleware.measurements["AddOneOperator"]
        + timer_middleware.measurements["AddTwoOperator"]
        < pipeline_time
    )
