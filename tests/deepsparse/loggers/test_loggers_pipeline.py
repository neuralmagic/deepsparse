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


from typing import Dict

import requests
from pydantic import BaseModel

from deepsparse import Pipeline
from deepsparse.loggers.logger_manager import LoggerManager
from deepsparse.operators import Operator
from deepsparse.routers import LinearRouter
from deepsparse.schedulers import OperatorScheduler


class IntSchema(BaseModel):
    value: int


class AddOneOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 1}


class AddTwoOperator(Operator):
    input_schema = IntSchema
    output_schema = IntSchema

    def run(self, inp: IntSchema, **kwargs) -> Dict:
        return {"value": inp.value + 2}


def test_pipeline_loggers():
    """basic logging test"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    system:
        "tag": # uses exact match. For regex, use "re:tag"
            - func: max
              freq: 1
              uses:
                - list

    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            self.logger_manager.log(
                value=1,
                tag="tag",
                log_type="system",
            )
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    assert len(list_logs) == 1


def test_pipeline_loggers_with_frequency():
    """one root logger, one tag, frequency of 2"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    system:
        "tag":
            - func: max
              freq: 2
              uses:
                - list

            - func: identity
              freq: 1
              uses:
                - list

    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            for _ in range(3):
                self.logger_manager.log(value=1, log_type="system", tag="tag")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    # three identity, one max
    assert len(list_logs) == 4

    # Check if there are three "[system.tag.identity]" and one "[system.tag.max]"
    identity_count = list_logs.count("[system.tag.identity]: 1")
    max_count = list_logs.count("[system.tag.max]: 1")

    # Check if there are three "[system.tag.identity]" and one "[system.tag.max]"
    assert (
        identity_count == 3
    ), "Expected three occurrences of '[system.tag.identity]: 1'"
    assert max_count == 1, "Expected one occurrence of '[system.tag.max]: 1'"


def test_pipeline_loggers_with_frequency_multiple_tags():
    """one logger multiple tag, exact match"""

    config = """
    loggers:
        default:
            name: PythonLogger
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger
    system:
        tag1:
        - func: identity
          freq: 1
          uses:
            - list
        - func: identity
          freq: 2
          uses:
            - list
        tag2:
        - func: identity
          freq: 1
          uses:
            - list
        - func: identity
          freq: 2
          uses:
            - list
    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):

            # record once
            self.logger_manager.system("one", tag="tag2", level="warning")

            # record twice
            self.logger_manager.system("two", tag="tag2", level="warning")

            # record once
            self.logger_manager.system("three", tag="tag2", level="warning")

            # record once
            self.logger_manager.system("four", tag="tag1", level="warning")

            # record twice
            self.logger_manager.system("five", tag="tag2", level="warning")

            # record once
            self.logger_manager.system("six", tag="tag2", level="warning")

            # record twice
            self.logger_manager.system("tag1 seven", tag="tag1", level="warning")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs
    # three identity, one max
    assert len(list_logs) == 10

    # Create a list of numbers to count
    numbers_to_count = ["one", "two", "three", "four", "five", "six"]

    # Initialize a dictionary to store counts
    number_counts = {}

    # Iterate through the list and count the numbers
    for number in numbers_to_count:
        count = sum(
            1 for tag in list_logs if f"[system.tag2.identity]: {number}" in tag
        )
        number_counts[number] = count

    # Check if the counts match the expected values
    for number in numbers_to_count:
        assert number_counts[number] == list_logs.count(
            f"[system.tag2.identity]: {number}"
        ), (
            f"Expected {number_counts[number]} occurrences of "
            "'[system.tag2.identity]: {number}'"
        )


def test_pipeline_loggers_with_two_log_types():
    # one logger multiple tag, regex
    """one root logger, one tag, frequency of 2"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    system:
        "tag":
            - func: max
              freq: 2
              uses:
                - list
            - func: identity
              freq: 1
              uses:
                - list
    metric:
        "tag":
            - func: max
              freq: 2
              uses:
                - list
            - func: identity
              freq: 1
              uses:
                - list
    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            for _ in range(3):
                self.logger_manager.log(value=1, log_type="system", tag="tag")
            for _ in range(3):
                self.logger_manager.log(value=1, log_type="metric", tag="tag")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    # three identity, one max
    assert len(list_logs) == 4 + 4

    # Check if there are three "[system.tag.identity]" and one "[system.tag.max]"
    system_identity_count = list_logs.count("[system.tag.identity]: 1")
    system_max_count = list_logs.count("[system.tag.max]: 1")

    # Check if there are three "[system.tag.identity]" and one "[system.tag.max]"
    assert (
        system_identity_count == 3
    ), "Expected three occurrences of '[system.tag.identity]: 1'"
    assert system_max_count == 1, "Expected one occurrence of '[system.tag.max]: 1'"

    metric_identity_count = list_logs.count("[metric.tag.identity]: 1")
    metric_max_count = list_logs.count("[metric.tag.max]: 1")

    # Check if there are three "[metric.tag.identity]" and one "[metric.tag.max]"
    assert (
        metric_identity_count == 3
    ), "Expected three occurrences of '[metric.tag.identity]: 1'"
    assert metric_max_count == 1, "Expected one occurrence of '[metric.tag.max]: 1'"


def test_pipeline_loggers_no_tag_match():
    """Skip logs if no tag match"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    system:
        "tag":
            - func: max
              freq: 2
              uses:
                - list

            - func: identity
              freq: 1
              uses:
                - list
    metric:
        "ta":
            - func: max
              freq: 2
              uses:
                - list

    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            for _ in range(3):
                self.logger_manager.log(value=1, log_type="system", tag="tag3")
                self.logger_manager.log(value=1, log_type="system", tag="ta")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    # tag shouldnt match with tag3, ta shouldnt match with tag3
    # and ta shouldnt match wit metric log
    assert len(list_logs) == 0


def test_pipeline_loggers_with_frequency_tags_multiple_capture():
    """Metric logger with config"""

    config = """
    loggers:
        list:
            name: tests/deepsparse/loggers/registry/loggers/list_logger.py:ListLogger

    metric:
        "tag": # uses exact match. For regex, use "re:tag"
            - func: max
              freq: 1
              uses:
                - list
              capture:
                - "re:.*" # capture all keys and class prop

    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            self.logger_manager.log(value=1, log_type="metric", tag="tag")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    list_logs = AddThreePipeline.logger_manager.leaf_logger["list"].logs

    assert len(list_logs) == 1


def test_pipeline_with_prometheus():
    """Test with prometheus logger"""

    config = """
    loggers:
        prom:
            name: PrometheusLogger

    performance:
        "tag": # uses exact match. For regex, use "re:tag"
            - func: max
              freq: 1
              uses:
                - prom

    """

    class LoggerPipeline(Pipeline):
        def __call__(self, *args, **kwargs):
            self.logger_manager.log(value=1, log_type="performance", tag="tag")
            return super().__call__(*args, **kwargs)

    AddThreePipeline = LoggerPipeline(
        ops=[AddOneOperator(), AddTwoOperator()],
        router=LinearRouter(end_route=2),
        schedulers=[OperatorScheduler()],
        logger_manager=LoggerManager(config),
    )

    pipeline_input = IntSchema(value=5)
    pipeline_output = AddThreePipeline(pipeline_input)

    assert pipeline_output.value == 8

    prometheus_logger = AddThreePipeline.logger_manager.leaf_logger["prom"]
    response = requests.get(f"http://0.0.0.0:{prometheus_logger.port}").text
    request_log_lines = response.split("\n")

    assert request_log_lines[-5] == 'deepsparse_tag_sum{pipeline_name="tag"} 1.0'
    assert request_log_lines[-6] == 'deepsparse_tag_count{pipeline_name="tag"} 1.0'
