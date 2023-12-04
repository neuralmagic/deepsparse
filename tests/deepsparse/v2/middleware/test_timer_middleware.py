# from tests.deepsparse.v2.middleware.utils import CounterMiddleware
# from deepsparse.v2.middleware.test_timer_middleware import TimerMiddleware

# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by call_nextlicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from typing import Any, Dict

from deepsparse.v2.middleware import TimerMiddleware
from deepsparse.v2.middleware.middlewares import (
    MiddlewareCallable,
    MiddlewareManager,
    MiddlewareSpec,
)
from deepsparse.v2.pipeline import PipelineMiddleware as Pipeline
from deepsparse.v2.routers import LinearRouter
from deepsparse.v2.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from tests.deepsparse.v2.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)


class PrintingMiddleware(MiddlewareCallable):
    def __init__(self, call_next: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        print(f"{self.identifier}: before call_next")
        result = self.call_next(*args, **kwargs)
        print(f"{self.identifier}: after call_next: {result}")
        return result


class SendStateMiddleware(MiddlewareCallable):
    def __init__(self, call_next: MiddlewareCallable, identifier: str):
        self.identifier: str = identifier
        self.call_next: MiddlewareCallable = call_next

    def __call__(self, *args, **kwargs) -> Any:
        name = self.__class__.__name__
        self.send(self.reducer, 0)

        result = self.call_next(*args, **kwargs)
        self.send(self.reducer, 1)

        return result

    def reducer(self, state: Dict, *args, **kwargs):
        name = self.__class__.__name__
        if name not in state:
            state[name] = []
        state[name].append(args[0])
        return state


def test_timer_middleware():
    """Check that timer gets saved in Pipeline and Ops"""

    middlewares = [
        MiddlewareSpec(PrintingMiddleware, identifier="A"),
        MiddlewareSpec(SendStateMiddleware, identifier="D"),
        MiddlewareSpec(TimerMiddleware, identifier="C"),
    ]

    middleware_manager = MiddlewareManager(middlewares)
    ops = [
        AddOneOperator(middleware_manager=middleware_manager),
        AddTwoOperator(middleware_manager=middleware_manager),
    ]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware_manager=middleware_manager,
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    state = AddThreePipeline.middleware_manager.state
    assert "measurements" in state
    assert "SendStateMiddleware" in state

    measurements = state["measurements"]

    # three measurements, two operators + one pipeline
    assert len(measurements) == len(ops) + 1

    # assert pipeline time is more than the sum of two ops
    op_time1, op_time_2 = measurements[0], measurements[1]
    pipeline_time = measurements[-1]
    assert pipeline_time > op_time1 + op_time_2

    # SendStateMiddleware, order of calls:
    # Pipeline start, AddOneOperator start, AddOneOperator end
    # AddTwoOperator start, AddTwoOperator end, Pipeline_ end
    expected_order = [0, 0, 1, 0, 1, 1]
    assert state["SendStateMiddleware"] == expected_order
