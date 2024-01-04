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


from collections import defaultdict
from typing import List

from deepsparse.middlewares import MiddlewareManager, MiddlewareSpec, TimerMiddleware
from deepsparse.pipeline import Pipeline
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from deepsparse.utils.state import InferenceState
from tests.deepsparse.middlewares import PrintingMiddleware, SendStateMiddleware
from tests.deepsparse.pipelines.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)
from tests.deepsparse.utils.wrappers import asyncio_run


def test_timer_middleware_timings_saved_in_timer_manager():
    """Check runtimes from timer manager saved into timer_manager"""

    middlewares = [
        MiddlewareSpec(PrintingMiddleware),  # debugging
        MiddlewareSpec(SendStateMiddleware),  # for callable entry and exit order
        MiddlewareSpec(TimerMiddleware),  # for timer
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

    pipeline_measurements: List[
        defaultdict
    ] = AddThreePipeline.timer_manager.measurements
    measurements = pipeline_measurements[0]

    # Pipeline, AddOneOperator, AddTwoOperator should have one measurement each
    assert len(measurements) == len(ops) + 1

    # assert pipeline time is more than the sum of two ops
    pipeline_time: List[float] = measurements["total"]
    add_one_operator_time, add_two_operator_time = (
        measurements["AddOneOperator"],
        measurements["AddTwoOperator"],
    )

    assert pipeline_time > add_one_operator_time + add_two_operator_time

    # check middleware triggered for Pipeline and Ops as expected
    state = AddThreePipeline.middleware_manager.state
    assert "SendStateMiddleware" in state

    # SendStateMiddleware, order of calls:
    # Pipeline start, AddOneOperator start, AddOneOperator end
    # AddTwoOperator start, AddTwoOperator end, Pipeline end
    expected_order = [0, 0, 1, 0, 1, 1]
    assert state["SendStateMiddleware"] == expected_order


def test_middleware_nested_pipeline():
    middlewares = [
        MiddlewareSpec(PrintingMiddleware),  # debugging
        MiddlewareSpec(SendStateMiddleware),  # for callable entry and exit order
        MiddlewareSpec(TimerMiddleware),  # for timer
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

    inference_state = InferenceState()
    inference_state.create_state({})
    timer = AddThreePipeline.timer_manager.get_new_timer()
    inference_state.set_timer(timer)

    pipeline_output = AddThreePipeline(pipeline_input, inference_state=inference_state)

    assert pipeline_output.value == 8

    pipeline_measurements: List[
        defaultdict
    ] = AddThreePipeline.timer_manager.measurements
    measurements = pipeline_measurements[0]

    # Nested Pipeline measumenets not recorded, just
    # AddOneOperator, AddTwoOperator should have one measurement each
    assert len(measurements) == len(ops)

    assert "AddOneOperator" in measurements
    assert "AddTwoOperator" in measurements


def test_timer_middleware_shared_timer():
    """
    Check shared timer are saved an expected format and
    check that averages are displayed
    """
    middlewares = [
        MiddlewareSpec(TimerMiddleware),  # for timer
    ]

    ops = [AddOneOperator(), AddTwoOperator()]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        middleware_manager=MiddlewareManager(middlewares),
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8
    assert len(AddThreePipeline.timer_manager.measurements) == 1

    # share the timer manager
    shared_timer_manager = AddThreePipeline.timer_manager

    AddThreePipeline2 = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        middleware_manager=MiddlewareManager(middlewares),
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

    # Check that the avaerages exist and have correct values
    averages = AddThreePipeline.timer_manager.average()

    keys = ["AddOneOperator", "AddTwoOperator", "total"]
    for key in keys:
        time_combined = pipeline1_measuremnts[key] + pipeline2_measuremnts[key]
        assert averages["iteration"][key] == len(time_combined) / 2
        assert averages["time"][key] == sum(time_combined) / len(time_combined)


def test_text_generation_creation_pipeline_has_timer_measurements():
    """Check the text gen pipeline creation timings are saved"""
    from deepsparse import TextGeneration

    middlewares = [
        MiddlewareSpec(TimerMiddleware),  # for timer
    ]

    model = TextGeneration(
        model_path="hf:mgoin/TinyStories-1M-ds",
        middleware_manager=MiddlewareManager(middlewares),
    )

    prompt = "How to make banana bread?"
    generation_config = {"max_new_tokens": 100}

    text = model(prompt, **generation_config).generations[0].text
    assert text is not None
    measurements = model.timer_manager.measurements

    expected_keys = {
        "ParseTextGenerationInputs",
        "PrepareforPrefill",
        "AutoRegressiveOperatorPreprocess",
        "NLEngineOperator",
        "CompilePromptLogits",
        "PrepareGeneration",
        "GenerateNewTokenOperator",
        "CompileGeneratedTokens",
        "CompileGenerations",
        "JoinOutput",
        "ProcessOutputs",
        "total",
    }
    for key in measurements[0].keys():
        expected_keys.remove(key)
        assert len(measurements[0][key]) > 0

    assert len(expected_keys) == 0


@asyncio_run
async def test_timer_middleware_timings_saved_in_timer_manager_async():
    """Check middlewares in async_run"""

    middlewares = [
        MiddlewareSpec(PrintingMiddleware),  # debugging
        MiddlewareSpec(SendStateMiddleware),  # for callable entry and exit order
        MiddlewareSpec(TimerMiddleware),  # for timer
    ]

    ops = [AddOneOperator(), AddTwoOperator()]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware_manager=MiddlewareManager(middlewares),
    )

    inference_state = InferenceState()
    inference_state.create_state({})

    pipeline_input = IntSchema(value=5)

    pipeline_output = await AddThreePipeline.run_async(
        pipeline_input, inference_state=inference_state
    )

    assert pipeline_output.value == 8

    pipeline_measurements: List[
        defaultdict
    ] = AddThreePipeline.timer_manager.measurements
    measurements = pipeline_measurements[0]

    # Pipeline, AddOneOperator, AddTwoOperator should have one measurement each
    assert len(measurements) == len(ops) + 1

    # assert pipeline time is more than the sum of two ops
    pipeline_time: List[float] = measurements["total"]
    add_one_operator_time, add_two_operator_time = (
        measurements["AddOneOperator"],
        measurements["AddTwoOperator"],
    )

    assert pipeline_time > add_one_operator_time + add_two_operator_time
    # check middleware triggered for Pipeline and Ops as expected
    state = AddThreePipeline.middleware_manager.state
    assert "SendStateMiddleware" in state

    # SendStateMiddleware, order of calls:
    #  AddOneOperator start, AddOneOperator end
    # AddTwoOperator start, AddTwoOperator end
    expected_order = [0, 1, 0, 1]
    assert state["SendStateMiddleware"] == expected_order
