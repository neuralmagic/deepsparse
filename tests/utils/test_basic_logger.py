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

"""
Simple example and test of a dummy pipeline
"""

import time
from collections import defaultdict
from typing import Dict

from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.loggers_v2.logger_manager import LoggerManager
from deepsparse.operators import Operator
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import OperatorScheduler


class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        inference_state = kwargs.get("inference_state")
        with inference_state.time(id="AddOneOperator"):
            time.sleep(0.2)
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        inference_state = kwargs.get("inference_state")
        with inference_state.time(id="AddTwoOperator"):
            time.sleep(0.5)
        return {"value": inp.value + 2}


def test_pipeline_fine_grained_timer_record_operator_run_times():
    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers_v2/registry/loggers/list_logger.py:ListLogger

    metric:
        "re:.*":
            - func: identity
              freq: 1
              uses:
                - list
              capture:
                - "re:.*"
    """
    AddThreePipeline = Pipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )
    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    measurements: defaultdict[list] = AddThreePipeline.timer_manager.measurements[0]

    assert len(measurements) == 2
    expected_keys = {"AddTwoOperator", "AddOneOperator"}
    for key in measurements.keys():
        expected_keys.remove(key)
    assert len(expected_keys) == 0

    list_log = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    assert len(list_log) == 2
