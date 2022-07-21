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
from concurrent.futures import ThreadPoolExecutor

import pytest
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


@pytest.mark.parametrize("task", supported_tasks, scope="class")
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

    def test_pipeline_creation(self, dynamic_batch_pipeline):
        # Will fail if fixture request fails
        pass

    @pytest.mark.parametrize(
        "batch_size",
        [
            # 1,
            5,
            # 10,
        ],
    )
    def test_execution_with_multiple_batch_sizes(
        self,
        batch_size,
        dynamic_batch_pipeline,
    ):
        inputs = dynamic_batch_pipeline.input_schema.create_test_inputs(
            batch_size=batch_size,
        )

        outputs = dynamic_batch_pipeline(**inputs)

        assert outputs

    @pytest.mark.parametrize("batch_size", [10])
    def test_pipeline_call_is_blocking(
        self,
        dynamic_batch_pipeline,
        batch_size,
    ):
        inputs = dynamic_batch_pipeline.input_schema.create_test_inputs(
            batch_size=batch_size,
        )
        output = dynamic_batch_pipeline(**inputs)
        assert not isinstance(output, concurrent.futures.Future), (
            "Expected dynamic batch pipeline to be blocking but got"
            "got a concurrent.futures.Future object instead"
        )

        assert type(output) == dynamic_batch_pipeline.output_schema

    @pytest.mark.parametrize(
        "batch_size",
        [
            1,
            2,
            10,
        ],
    )
    def test_order_retention_against_static_batch(
        self, task, executor, batch_size, dynamic_batch_pipeline
    ):
        inputs = Pipeline.create(
            task=task,
        ).input_schema.create_test_inputs(batch_size)

        # Run each sample through its own pipeline
        static_batch_threaded_pipeline = Pipeline.create(
            task=task,
            batch_size=batch_size,
            executor=executor,
        )
        static_outputs = static_batch_threaded_pipeline(**inputs).result()
        dynamic_outputs = dynamic_batch_pipeline(**inputs)

        # Check that order is maintained
        assert static_outputs == dynamic_outputs
