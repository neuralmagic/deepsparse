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
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy

import pytest
from deepsparse.pipeline import Pipeline, _initialize_executor_and_workers
from deepsparse.timing import InferenceTimingSchema
from tests.utils import mock_engine


def test_split_engine_inputs():
    inp = [numpy.zeros((4, 28)) for _ in range(3)]

    out = Pipeline.split_engine_inputs(inp, batch_size=4)
    assert numpy.array(out).shape == (1, 3, 4, 28)

    out = Pipeline.split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    out = Pipeline.split_engine_inputs(inp, batch_size=1)
    assert numpy.array(out).shape == (4, 3, 1, 28)


def test_join_opposite_of_split():
    inp = [numpy.random.rand(4, 28) for _ in range(3)]

    out = Pipeline.split_engine_inputs(inp, batch_size=2)
    assert numpy.array(out).shape == (2, 3, 2, 28)

    joined = Pipeline.join_engine_outputs(out)
    assert numpy.array(joined).shape == (3, 4, 28)

    for i, j in zip(inp, joined):
        assert (i == j).all()


def test_split_engine_inputs_uneven_raises_error():
    with pytest.raises(
        RuntimeError,
        match=(
            "batch size of 3 passed into pipeline "
            "is not divisible by model batch size of 2"
        ),
    ):
        Pipeline.split_engine_inputs([numpy.zeros((3, 28))], batch_size=2)


@mock_engine(rng_seed=0)
def test_split_interaction_with_forward_batch_size_1(engine_mock):
    pipeline = Pipeline.create("token_classification", batch_size=1)
    with mock.patch.object(
        Pipeline, "engine_forward", wraps=pipeline.engine_forward
    ) as engine_forward:
        pipeline("word")
        assert engine_forward.call_count == 1

        pipeline("two words".split())
        assert engine_forward.call_count == 3

        pipeline("two words for me".split())
        assert engine_forward.call_count == 7


@mock_engine(rng_seed=0)
def test_split_interaction_with_forward_batch_size_2(engine_forward):
    pipeline = Pipeline.create("token_classification", batch_size=2)
    with mock.patch.object(
        Pipeline, "engine_forward", wraps=pipeline.engine_forward
    ) as engine_forward:
        with pytest.raises(RuntimeError, match="is not divisible"):
            pipeline("word")

        pipeline("two words".split())
        assert engine_forward.call_count == 1

        pipeline("two words for me".split())
        assert engine_forward.call_count == 3


def test_pipeline_executor_num_workers():
    executor, _ = _initialize_executor_and_workers(2, None)
    assert executor._max_workers == 1

    executor, _ = _initialize_executor_and_workers(2, 2)
    assert executor._max_workers == 2

    executor, _ = _initialize_executor_and_workers(None, 2)
    assert executor._max_workers == 2

    executor, _ = _initialize_executor_and_workers(None, ThreadPoolExecutor(3))
    assert executor._max_workers == 3

    executor, _ = _initialize_executor_and_workers(1, ThreadPoolExecutor(3))
    assert executor._max_workers == 3

    executor, _ = _initialize_executor_and_workers(None, None)
    assert executor._max_workers >= 1


@mock_engine(rng_seed=0)
def test_pipeline_call_is_async(engine_mock):
    # here we make engine_forward take a different amount of time
    # based on the input. the total time should be the longest
    # forward call, instead of sum.
    #
    # The time each forward takes is just summing up character counts
    # in each word. So "aaaa" will sleep for 50ms whereas "a" will sleep for 10ms
    executor = ThreadPoolExecutor(max_workers=1)
    pipeline = Pipeline.create("token_classification", batch_size=1, executor=executor)

    def sleep_for_10ms(xs):
        time.sleep(10 / 1000)
        return pipeline.engine(xs)

    with mock.patch.object(Pipeline, "engine_forward", side_effect=sleep_for_10ms):
        start = time.time_ns()
        pipeline(["abcdef"] * 6)
        end = time.time_ns()
        dur_ms = (end - start) * 1e-6
        assert abs(dur_ms - 60) < 10

        pipeline.executor = ThreadPoolExecutor(max_workers=2)

        start = time.time_ns()
        pipeline(["abcdef"] * 6)
        end = time.time_ns()
        dur_ms = (end - start) * 1e-6
        assert abs(dur_ms - 30) < 10


@mock_engine(rng_seed=0)
def test_run_with_monitoring(engine_mock):
    pipeline = Pipeline.create("token_classification", batch_size=1)
    _, _, _, inference_timing = pipeline.run_with_monitoring(
        "all_your_base_are_belong_to_us"
    )
    assert isinstance(inference_timing, InferenceTimingSchema)
    assert (
        pytest.approx(
            inference_timing.pre_process
            + inference_timing.post_process
            + inference_timing.engine_forward,
            10e-3,
        )
        == inference_timing.total_inference
    )
