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


from deepsparse.middlewares import MiddlewareManager, MiddlewareSpec
from deepsparse.pipeline import Pipeline
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from tests.deepsparse.middlewares import PrintingMiddleware, SendStateMiddleware
from tests.deepsparse.pipelines.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)


def test_pipeline_with_middleware():
    """Check runtimes from timer manager saved into timer_manager"""

    middlewares = [
        MiddlewareSpec(PrintingMiddleware),  # debugging
        MiddlewareSpec(SendStateMiddleware),  # for callable entry and exit order
    ]

    ops = [AddOneOperator(), AddTwoOperator()]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware_manager=MiddlewareManager(middlewares),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    # check middleware triggered for Pipeline and Ops as expected
    state = AddThreePipeline.middleware_manager.state
    assert "SendStateMiddleware" in state

    # SendStateMiddleware, order of calls:
    # Pipeline start, AddOneOperator start, AddOneOperator end
    # AddTwoOperator start, AddTwoOperator end, Pipeline end
    expected_order = [0, 0, 1, 0, 1, 1]
    assert state["SendStateMiddleware"] == expected_order
