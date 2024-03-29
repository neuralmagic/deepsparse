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
from deepsparse import TextGeneration
from deepsparse.loggers.logger_manager import LoggerManager
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


PROMPT = "How to make banana bread?"
GENERATION_CONFIG = {"max_new_tokens": 10}


@pytest.fixture
def text_generation_instance(frequency: int = 1):
    config = f"""
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    metric:
        "re:.*":  # regex match all
            - func: identity
              freq: {frequency}
              uses:
                - list

    """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),
    ]

    model = TextGeneration(
        model_path="hf:mgoin/TinyStories-1M-ds",
        middleware_manager=MiddlewareManager(middlewares),
        logger_manager=LoggerManager(config),
    )

    text = model(PROMPT, **GENERATION_CONFIG).generations[0].text

    # wait for async loggers to finish
    model.logger_manager.wait_for_completion()

    assert text is not None

    return model


def test_logger_middleware_logs_saved_in_list_logger():
    """Check metric logs in LoggerMiddleware are logged as expected"""
    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    metric:
        "re:(?i)operator": # regex match with non case sensitive Operator
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

    AddThreePipeline.logger_manager.wait_for_completion()

    # check list logger logs
    list_log = AddThreePipeline.logger_manager.leaf_logger["list"].logs
    assert len(list_log) == 2

    expected_logs = set(
        [
            "[metric.AddOneOperator.identity]: value=6",
            "[metric.AddTwoOperator.identity]: value=8",
        ]
    )
    for tag in list_log:
        expected_logs.remove(tag)
    assert len(expected_logs) == 0


@pytest.mark.parametrize(
    "frequency",
    [
        2,
        3,
        4,
    ],
)
def test_text_generation_pipeline_trigger_logger_with_run_time_with_frequency_filter(
    frequency, text_generation_instance
):
    """Check logger with frequency filter and timer middleware"""

    config = f"""
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    metric:
        "re:.*":  # regex match all
            - func: identity
              freq: {frequency}
              uses:
                - list

    """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),
    ]

    model = TextGeneration(
        model_path="hf:mgoin/TinyStories-1M-ds",
        middleware_manager=MiddlewareManager(middlewares),
        logger_manager=LoggerManager(config),
    )

    text = model(PROMPT, **GENERATION_CONFIG).generations[0].text

    # wait for async loggers to finish
    model.logger_manager.wait_for_completion()

    assert text is not None
    list_log = model.logger_manager.leaf_logger["list"].logs

    max_expected_len_list_logs = (
        len(text_generation_instance.logger_manager.leaf_logger["list"].logs)
        / frequency
    )
    assert math.floor(len(list_log)) <= math.floor(max_expected_len_list_logs)


@asyncio_run
async def test_timer_middleware_loggings_and_timings_async():
    """Check middlewares in async_run using timer and logger"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    metric:
        "re:.*":
            - func: identity
              freq: 1
              uses:
                - list
              capture:
                - "re:.*"
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
    list_log = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    # wait for submitted jobs to complete
    AddThreePipeline.logger_manager.wait_for_completion()

    # two logs and one timer
    assert len(list_log) == 3
