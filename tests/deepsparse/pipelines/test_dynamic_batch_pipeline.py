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
import itertools
from concurrent.futures import ThreadPoolExecutor

import numpy

import pytest
from data_helpers import create_test_inputs
from deepsparse import Pipeline


@pytest.fixture()
def executor():
    """
    An Auto-delete fixture for yielding a concurrent.futures.ThreadPoolExecutor()
    object, uses cpu_count + 4 threads, limited 32 to avoid consuming surprisingly
    large resource on many core machines
    """
    yield ThreadPoolExecutor()


supported_tasks = [
    "text_classification",
    "token_classification",
    "yolo",
    "image_classification",
]

batch_sizes = [
    1,
    2,
    10,
]

tasks_batch_size_pairs = list(itertools.product(supported_tasks, batch_sizes))


def compare(expected, actual):
    assert type(expected) == type(actual)

    if isinstance(expected, (list, float, numpy.ndarray)):
        expected_np = numpy.asarray(expected)
        actual_np = numpy.asarray(actual)
        assert numpy.allclose(expected_np, actual_np, rtol=1e-3)
    elif isinstance(expected, dict):
        assert list(expected.keys()).sort() == list(actual.keys()).sort()
        for key, exp_value in expected.items():
            assert compare(exp_value, actual[key])
    else:
        assert expected == actual
    return True


@pytest.mark.parametrize("task, batch_size", tasks_batch_size_pairs, scope="class")
class TestDynamicBatchPipeline:
    @pytest.fixture()
    def dynamic_batch_pipeline(self, task, executor):
        """
        An auto-delete fixture to yield a Dynamic Batch Pipeline, this Pipeline
        is capable of executing different batch sizes.
        """
        yield Pipeline.create(
            task=task,
            batch_size=None,
            executor=executor,
        )

    @pytest.fixture()
    def static_batch_pipeline(self, task, executor, batch_size):
        """
        An auto-delete fixture to yield a Static Batch Pipeline
        """
        assert batch_size is not None
        yield Pipeline.create(
            task=task,
            batch_size=batch_size,
            executor=executor,
        )

    @pytest.fixture()
    def inputs(self, task, batch_size):
        """
        An auto-delete fixture to get task inputs
        """
        yield create_test_inputs(task=task, batch_size=batch_size)

    @pytest.fixture()
    def dynamic_batch_outputs(self, inputs, dynamic_batch_pipeline):
        """
        An auto-delete fixture to yield output from dynamic batch pipline
        """
        yield dynamic_batch_pipeline(**inputs)

    @pytest.fixture()
    def static_batch_outputs(self, inputs, static_batch_pipeline):
        """
        An auto-delete fixture to yield output from dynamic batch pipline
        """
        results = static_batch_pipeline(**inputs)
        if isinstance(results, concurrent.futures.Future):
            yield results.result()
        else:
            yield results

    def test_pipeline_creation(self, batch_size, dynamic_batch_pipeline):
        # Will fail if fixture request fails
        pass

    def test_execution_and_output(
        self,
        dynamic_batch_outputs,
        dynamic_batch_pipeline,
    ):
        assert dynamic_batch_outputs
        assert not isinstance(dynamic_batch_outputs, concurrent.futures.Future), (
            "Expected dynamic batch pipeline to be blocking but got"
            "got a concurrent.futures.Future object instead"
        )
        assert type(dynamic_batch_outputs) == dynamic_batch_pipeline.output_schema

    def test_order_retention_against_static_batch(
        self,
        static_batch_outputs,
        dynamic_batch_outputs,
    ):
        expected_dict = static_batch_outputs.dict()
        actual_dict = dynamic_batch_outputs.dict()

        # Check that order is maintained
        assert static_batch_outputs == dynamic_batch_outputs or compare(
            expected_dict, actual_dict
        )
