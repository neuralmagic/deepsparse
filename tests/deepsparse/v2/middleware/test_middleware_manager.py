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


from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from tests.deepsparse.v2.middleware.utils import CounterMiddleware, OpsTrackerMiddleware
from tests.deepsparse.v2.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)


def test_middleware_triggered_in_expected_order_for_ops():
    """Check that when ops gets triggered, middleware behaves as expected"""
    op_track_middleware = OpsTrackerMiddleware()
    counter_middleware = CounterMiddleware()
    middleware = [op_track_middleware, counter_middleware]
    ops = [AddOneOperator(), AddTwoOperator()]
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

    expected_middleware_start_order = ["AddOneOperator", "AddTwoOperator"]
    actual_middleware_start_order = AddThreePipeline._middleware.middleware[
        0
    ].start_order

    # check pass my reference for middleware
    assert op_track_middleware == AddThreePipeline._middleware.middleware[0]
    assert counter_middleware == AddThreePipeline._middleware.middleware[1]

    actual_middleware_end_order = AddThreePipeline._middleware.middleware[0].start_order

    assert (
        expected_middleware_start_order
        == actual_middleware_start_order
        == actual_middleware_end_order
    )

    assert counter_middleware.start_called == counter_middleware.end_called == len(ops)
