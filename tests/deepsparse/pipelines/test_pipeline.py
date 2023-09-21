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

import flaky
import pytest
from deepsparse.base_pipeline import BasePipeline
from deepsparse.pipeline import (
    Pipeline,
    PipelineConfig,
    _initialize_executor_and_workers,
)
from tests.utils import mock_engine


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
        # this is okay because we can pad batches
        pipeline("word")
        assert engine_forward.call_count == 1

        pipeline("two words".split())
        assert engine_forward.call_count == 2

        pipeline("two words for me".split())
        assert engine_forward.call_count == 4


@pytest.fixture
def base_pipeline_example():
    @BasePipeline.register(task="base_example")
    class BasePipelineExample(BasePipeline):
        def __init__(self, base_specific, **kwargs):
            self._base_specific = base_specific
            super().__init__(**kwargs)

        def __call__(self, *args, **kwargs):
            pass

        def input_schema(self):
            pass

        def output_schema(self):
            pass

        @property
        def base_specific(self):
            return self._base_specific

    kwargs = {"base_specific": "base_specific"}
    base_pipeline = BasePipeline.create(
        task="base_example", alias="base_alias", **kwargs
    )
    return base_pipeline, BasePipelineExample, kwargs


def test_base_pipeline(base_pipeline_example):
    base_pipeline = base_pipeline_example[0]
    pipeline = base_pipeline_example[1]
    kwargs = base_pipeline_example[-1]

    assert base_pipeline.base_specific == kwargs["base_specific"]

    cls = BasePipeline._get_task_constructor("base_example")
    assert cls == pipeline

    config = base_pipeline.to_config()
    assert isinstance(config, PipelineConfig)
    assert config.kwargs["base_specific"] == base_pipeline.base_specific


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


@flaky.flaky(max_runs=2, min_passes=1)
@mock_engine(rng_seed=0)
def test_pipeline_call_is_async(engine_mock):
    # attempts to verify that pipeline calls to engine are async
    # by executing at timing test with different number of workers

    executor = ThreadPoolExecutor(max_workers=1)
    pipeline = Pipeline.create("token_classification", batch_size=1, executor=executor)

    def sleep_then_engine_forward(xs, context):
        # each call to engine_forward also sleeps
        time.sleep(50 / 1000)
        return pipeline.engine(xs)

    with mock.patch.object(
        Pipeline, "engine_forward", side_effect=sleep_then_engine_forward
    ):
        start = time.perf_counter()
        # since there are 6 entries in the input, should sleep for (6 * 50) ms
        # since there are 1 worker threads, should take a total of (300 / 1) ms
        pipeline(["abcdef"] * 6)
        end = time.perf_counter()
        dur_1_worker = (end - start) * 1e3

        pipeline.executor = ThreadPoolExecutor(max_workers=2)

        # since there are 6 entries in the input, should sleep for (6 * 50) ms
        # since there are 2 worker threads, should take a total of (300 / 2) ms
        start = time.perf_counter()
        pipeline(["abcdef"] * 6)
        end = time.perf_counter()
        dur_2_worker = (end - start) * 1e3

        # instead of doing a hard comparison of timing for each separate
        # duration, do relative comparison of timing
        assert numpy.allclose(dur_1_worker / dur_2_worker, 2, atol=0.2)
