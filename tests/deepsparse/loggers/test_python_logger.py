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


import pytest
from deepsparse import MetricCategories, Pipeline, PythonLogger
from tests.utils import mock_engine


@pytest.mark.parametrize(
    "pipeline_name",
    [
        ("python_logger"),
        (None),
    ],
)
@mock_engine(rng_seed=0)
def test_python_logger(engine, pipeline_name, capsys):
    if pipeline_name:
        pipeline = Pipeline.create(
            "token_classification",
            batch_size=1,
            alias=pipeline_name,
            logger=PythonLogger(),
        )
    else:
        pipeline = Pipeline.create(
            "token_classification", batch_size=1, logger=PythonLogger()
        )
        pipeline_name = pipeline.task
    pipeline("all_your_base_are_belong_to_us")
    messages = capsys.readouterr().out.split("\n")
    relevant_logs = [message for message in messages if pipeline_name in message]
    assert len(relevant_logs) == 8
    assert (
        len(
            [
                log
                for log in relevant_logs
                if f"Category: {MetricCategories.SYSTEM.value}" in log
            ]
        )
        == 4
    )
    assert (
        len(
            [
                log
                for log in relevant_logs
                if f"Category: {MetricCategories.DATA.value}" in log
            ]
        )
        == 4
    )
    assert all(f"Identifier: {pipeline_name}" in log for log in relevant_logs)
