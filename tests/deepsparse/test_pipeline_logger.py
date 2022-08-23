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

from deepsparse import Pipeline
from deepsparse.pipeline_logger import LoggerManager, PipelineLogger


class TestPipelineLogger(PipelineLogger):
    def __init__(self, identifier):
        super().__init__(identifier=identifier)

    def log_data(self):
        pass

    def log_latency(self):
        pass


def test_logger_manager_single_logger():
    _test_logger_manager_(logger=TestPipelineLogger("test_logger1"))


def test_logger_manager_multiple_loggers():
    _test_logger_manager_(
        logger=[TestPipelineLogger("test_logger1"), TestPipelineLogger("test_logger2")]
    )


def _test_logger_manager_(logger, task_name="image_classification"):
    pipeline = Pipeline.create(task_name)
    logger_manager = LoggerManager.from_pipeline(pipeline)
    logger_manager.add(logger)
    assert (
        len(logger_manager.loggers) == 1
        if not isinstance(logger, list)
        else len(logger)
    )
    assert logger_manager.pipeline_name == task_name
    for logger_id, logger in logger_manager.loggers.items():
        logger.pipeline_name == task_name
