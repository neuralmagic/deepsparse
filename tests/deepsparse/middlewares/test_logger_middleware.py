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

from deepsparse.loggers_v2.logger_manager import LoggerManager
from deepsparse.middlewares import LoggerMiddleware, MiddlewareManager, MiddlewareSpec
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


# from tests.deepsparse.utils.wrappers import asyncio_run



# def test_logger_middleware_logs_saved_in_list_logger():
#     """Check metric logs in LoggerMiddleware logged as expected"""
#     config = """
#     logger:
#         list:
#             name: ListLogger

#     metric:
#         "(?i)operator":
#             - func: identity
#               freq: 1
#               uses:
#               - list
#     """

#     middlewares = [
#         MiddlewareSpec(LoggerMiddleware),  # for timer
#     ]

#     ops = [AddOneOperator(), AddTwoOperator()]

#     AddThreePipeline = Pipeline(
#         ops=ops,
#         router=LinearRouter(end_route=2),
#         schedulers=[OperatorScheduler()],
#         continuous_batching_scheduler=ContinuousBatchingScheduler,
#         middleware_manager=MiddlewareManager(middlewares),
#         logger_manager=LoggerManager(config),
#     )

#     pipeline_input = IntSchema(value=5)
#     pipeline_output = AddThreePipeline(pipeline_input)
#     assert pipeline_output.value == 8
    
#     # check list logger logs
#     list_log = AddThreePipeline.logger_manager.metric.logger["(?i)operator"][0].logs
#     assert len(list_log) == 2
    
#     expected_logs = set(
#         ['metric.AddOneOperator.6.identity', 'metric.AddTwoOperator.8.identity']
#     )
#     for tag in list_log:
#         expected_logs.remove(tag)
#     assert len(expected_logs) == 0

# (Pdb) 1146

def test_text_generation_creation_pipeline_has_timer_measurements():
    """Check the text gen pipeline creation timings are saved"""
    from deepsparse import TextGeneration
    config = """
    logger:
        list:
            name: ListLogger

    metric:
        # "(?i)operator\/([^)]*\/))":
        "(?i)operator([^)]*)":
        # ParseTextGenerationInputs:
            - func: max
              freq: 2
              uses:
                - list
            #   capture: 
            #     - max_new_tokens
        """

    middlewares = [
        MiddlewareSpec(LoggerMiddleware),  # for timer
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
    # breakpoint()
    # list_log = model.logger_manager.metric.logger["ParseTextGenerationInputs"][0].logs
    list_log = model.logger_manager.metric.leaf_logger["list"].logs
    
    print(list_log)
    breakpoint()
   

# @asyncio_run
# async def test_timer_middleware_timings_saved_in_timer_manager_async():
#     """Check middlewares in async_run"""

#     middlewares = [
#         MiddlewareSpec(PrintingMiddleware),  # debugging
#         MiddlewareSpec(SendStateMiddleware),  # for callable entry and exit order
#         MiddlewareSpec(TimerMiddleware),  # for timer
#     ]

#     ops = [AddOneOperator(), AddTwoOperator()]

#     AddThreePipeline = Pipeline(
#         ops=ops,
#         router=LinearRouter(end_route=2),
#         schedulers=[OperatorScheduler()],
#         continuous_batching_scheduler=ContinuousBatchingScheduler,
#         middleware_manager=MiddlewareManager(middlewares),
#     )

#     inference_state = InferenceState()
#     inference_state.create_state({})

#     pipeline_input = IntSchema(value=5)

#     pipeline_output = await AddThreePipeline.run_async(
#         pipeline_input, inference_state=inference_state
#     )

#     assert pipeline_output.value == 8

#     pipeline_measurements: List[
#         defaultdict
#     ] = AddThreePipeline.timer_manager.measurements
#     measurements = pipeline_measurements[0]

#     # Pipeline, AddOneOperator, AddTwoOperator should have one measurement each
#     assert len(measurements) == len(ops) + 1

#     # assert pipeline time is more than the sum of two ops
#     pipeline_time: List[float] = measurements["total_inference"]
#     add_one_operator_time, add_two_operator_time = (
#         measurements["AddOneOperator"],
#         measurements["AddTwoOperator"],
#     )

#     assert pipeline_time > add_one_operator_time + add_two_operator_time
#     # check middleware triggered for Pipeline and Ops as expected
#     state = AddThreePipeline.middleware_manager.state
#     assert "SendStateMiddleware" in state

#     # SendStateMiddleware, order of calls:
#     #  AddOneOperator start, AddOneOperator end
#     # AddTwoOperator start, AddTwoOperator end
#     expected_order = [0, 1, 0, 1]
#     assert state["SendStateMiddleware"] == expected_order
