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

import concurrent.futures
import time

import pytest
from deepsparse import Pipeline, ThreadPool


@pytest.mark.parametrize(
    "question,context,n",
    [
        ("who is mark", "mark is batman", 5),
        ("Fourth of July is Independence day", "when is Independence day", 5),
    ],
)
def test_qa_async_pipeline_is_faster(question, context, n):
    sync_qa_pipeline = Pipeline.create(task="question-answering")
    sequential_start_time = time.time()

    for _ in range(n):
        sync_qa_pipeline(question=question, context=context)

    avg_sequential_run_time = (time.time() - sequential_start_time) / n
    del sync_qa_pipeline

    futures = []
    async_qa_pipeline = Pipeline.create(
        task="question-answering",
        threadpool=ThreadPool(),
    )
    async_start_time = time.time()

    for _ in range(n):
        futures.append(async_qa_pipeline(question=question, context=context))

    concurrent.futures.wait(futures)  # returns when all futures complete
    avg_async_run_time = (time.time() - async_start_time) / n

    del async_qa_pipeline

    print(
        f"Sync: {avg_sequential_run_time}, Async: {avg_async_run_time}",
    )

    assert avg_async_run_time < avg_sequential_run_time, (
        "Asynchronous pipeline inference is slower than synchronous execution",
    )
