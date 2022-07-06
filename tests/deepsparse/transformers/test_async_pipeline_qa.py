import time
from deepsparse import Pipeline, ThreadPool
import concurrent.futures
import pytest


@pytest.mark.parametrize(
    "question,context",
    [
        ("who is mark", "mark is batman"),
        ("Fourth of July is Independence day", "when is Independence day"),
    ],
)
def test_qa_async_pipeline_is_faster(question, context):
    n = 10

    print("Without Threadpool ... ")
    qa_pipeline = Pipeline.create(task="question-answering")
    sequential_start_time = time.time()

    for _ in range(n):
        qa_pipeline(question=question, context=context)

    sequential_run_time = (time.time() - sequential_start_time) / n

    del qa_pipeline
    print("With ThreadPool ... ")
    futures = []
    qa_pipeline = Pipeline.create(
        task="question-answering",
        threadpool=ThreadPool(),
    )
    async_start_time = time.time()

    for _ in range(n):
        futures.append(
            qa_pipeline.submit(question=question, context=context)
        )

    concurrent.futures.wait(futures)
    async_run_time = (time.time() - async_start_time) / n
    print(f"Sync: {sequential_run_time}, Async: {async_run_time}",)

    assert async_run_time < sequential_run_time, (
        "Asynchronous pipeline inference is slower than synchronous execution",
    )
