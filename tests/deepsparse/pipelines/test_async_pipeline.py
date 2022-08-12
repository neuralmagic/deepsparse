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

import time
from concurrent.futures import Future, ThreadPoolExecutor

import pytest
from deepsparse import Pipeline


@pytest.fixture(scope="module")
def pipeline():
    """
    Auto-del fixture for Sequential Pipeline
    """
    yield Pipeline.create(task="question-answering")


@pytest.fixture(scope="module")
def executor():
    """
    Auto-del fixture for yielding a ThreadPoolExecutor
    """
    yield ThreadPoolExecutor()


@pytest.fixture(scope="module")
def threaded_pipeline(executor):
    """
    Auto-del fixture for Threaded Pipeline
    """
    yield Pipeline.create(task="question-answering", executor=executor)


@pytest.fixture(scope="module")
def qa_input():
    """
    Auto-del fixture that yield a valid input for a qa Pipeline
    """
    yield {
        "question": "Who am I?",
        "context": "I am Snorlax",
    }


@pytest.mark.parametrize(
    "tries",
    [
        1,
        2,
    ],
)
def test_async_submit_is_faster_than_sequential_execution(
    pipeline,
    threaded_pipeline,
    qa_input,
    tries,
):
    sequential_start_time = time.time()
    for _ in range(tries):
        pipeline(**qa_input)
    total_sequential_execution_time = time.time() - sequential_start_time

    async_start_time = time.time()
    for _ in range(tries):
        threaded_pipeline(**qa_input)
    async_submit_time = time.time() - async_start_time

    assert (
        async_submit_time < total_sequential_execution_time
    ), "Asynchronous pipeline submit is slower than sequential execution"


def test_async_pipeline_returns_a_future(threaded_pipeline, qa_input):
    return_value = threaded_pipeline(**qa_input)
    assert isinstance(return_value, Future), "Async Pipeline must return a Future"


def test_async_pipeline_results_in_correct_schema(
    pipeline,
    threaded_pipeline,
    qa_input,
):
    sync_result = pipeline(**qa_input)
    async_result = threaded_pipeline(**qa_input).result()

    assert type(sync_result) == type(
        async_result
    ), "Schema mismatch b/w Sequential and Async pipeline results"


def test_passing_threadpool_during_pipeline_call_returns_a_future(
    pipeline,
    qa_input,
    executor,
):
    return_value = pipeline(**qa_input, executor=executor)
    assert isinstance(
        return_value, Future
    ), "Sync Pipeline must return a Future when a executor is passed during call"


def test_passing_threadpool_during_pipeline_call_results_in_correct_schema(
    pipeline,
    qa_input,
    executor,
):
    sync_result = pipeline(**qa_input)
    async_result = pipeline(**qa_input, executor=executor).result()

    assert type(sync_result) == type(
        async_result
    ), "Schema mismatch b/w Sequential and Async pipeline"
