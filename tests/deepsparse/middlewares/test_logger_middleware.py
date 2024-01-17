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


import math
from collections import defaultdict
from typing import List

import pytest
from deepsparse.loggers_v2.logger_manager import LoggerManager
from deepsparse.middlewares import (
    LoggerMiddleware,
    MiddlewareManager,
    MiddlewareSpec,
    TimerMiddleware,
)
from deepsparse.pipeline import Pipeline
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from deepsparse.utils.state import InferenceState
from tests.deepsparse.pipelines.test_basic_pipeline import (
    AddOneOperator,
    AddTwoOperator,
    IntSchema,
)
from tests.deepsparse.utils.wrappers import asyncio_run


def test_logger_middleware_logs_saved_in_list_logger():
    """Check metric logs in LoggerMiddleware are logged as expected"""
    config = """
    logger:
        list:
            name: ListLogger

    metric:
        "(?i)operator":
            - func: identity
              freq: 1
              uses:
              - list
    """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),  # for timer
    ]

    ops = [AddOneOperator(), AddTwoOperator()]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware_manager=MiddlewareManager(middlewares),
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)
    assert pipeline_output.value == 8

    # check list logger logs
    list_log = AddThreePipeline.logger_manager.metric.logger["(?i)operator"][0].logs
    assert len(list_log) == 2

    expected_logs = set(
        [
            "[metric.AddOneOperator.identity] identity(AddOneOperator.value): 6",
            "[metric.AddTwoOperator.identity] identity(AddTwoOperator.value): 8",
        ]
    )
    for tag in list_log:
        expected_logs.remove(tag)
    assert len(expected_logs) == 0


@pytest.mark.parametrize(
    "frequency, max_expected_len_list_logs",
    [
        (1, 1836),  # middlewares/timer_middleware.py(34) - popping is_nested_key
        (2, 1836 / 2),
        (3, 1836 / 3),
        (4, 1836 / 4),
    ],
)
def test_text_generation_pipeline_trigger_logger_with_run_time_with_frequency_filter(
    frequency, max_expected_len_list_logs
):
    """Check logger with frequency filter and timer middleware"""

    from deepsparse import TextGeneration

    config = f"""
    logger:
        list:
            name: ListLogger

    metric:
        ".*":
            - func: max
              freq: {frequency}
              uses:
                - list
              capture:
                - ".*"
    """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),
        MiddlewareSpec(TimerMiddleware),  # for timer
    ]

    model = TextGeneration(
        model_path="hf:mgoin/TinyStories-1M-ds",
        middleware_manager=MiddlewareManager(middlewares),
        logger_manager=LoggerManager(config),
    )

    prompt = "How to make banana bread?"
    generation_config = {"max_new_tokens": 10}

    text = model(prompt, **generation_config).generations[0].text
    assert text is not None

    list_log = model.logger_manager.metric.leaf_logger["list"].logs

    len_attr = len(model.logger_manager.metric.counter)

    # Expected length of list_logger:
    # In frequency counter, if we have attrs {a: Na, b: Nb, c: Nc},
    # The expected high bound is N_x / frequency with zero remainder
    # Expected low bound is all attr have frequency - 1 remainders
    max_remainder = len_attr * (frequency - 1)

    assert (
        math.floor(max_expected_len_list_logs - max_remainder)
        <= math.floor(len(list_log))
        <= math.floor(max_expected_len_list_logs)
    )

    # check that timer is in logs
    assert all("⏱️" in log for log in list_log)

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
        "total_inference",
    }
    for key in measurements[0].keys():
        expected_keys.remove(key)
        assert len(measurements[0][key]) > 0

    assert len(expected_keys) == 0


@asyncio_run
async def test_timer_middleware_loggings_and_timings_async():
    """Check middlewares in async_run using timer and logger"""

    config = """
    logger:
        list:
            name: ListLogger

    metric:
        ".*":
            - func: identity
              freq: 1
              uses:
                - list
              capture:
                - ".*"
    """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),
        MiddlewareSpec(TimerMiddleware),  # for timer
    ]

    ops = [AddOneOperator(), AddTwoOperator()]

    AddThreePipeline = Pipeline(
        ops=ops,
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        continuous_batching_scheduler=ContinuousBatchingScheduler,
        middleware_manager=MiddlewareManager(middlewares),
        logger_manager=LoggerManager(config),
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
    pipeline_time: List[float] = measurements["total_inference"]
    add_one_operator_time, add_two_operator_time = (
        measurements["AddOneOperator"],
        measurements["AddTwoOperator"],
    )

    assert pipeline_time > add_one_operator_time + add_two_operator_time

    # check list logger logs
    list_log = AddThreePipeline.logger_manager.metric.leaf_logger["list"].logs
    assert len(list_log) == 2
    assert all("⏱️" in log for log in list_log)
