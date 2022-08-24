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

from concurrent.futures import ThreadPoolExecutor
from deepsparse import Pipeline
import pytest
from tests.deepsparse.pipelines.test_pipeline import FakePipeline
from .data_helpers import create_test_inputs

_SUPPORTED_TASKS = [
    "text_classification",
    "token_classification",
    "yolo",
    "image_classification",
    "yolact",
]

@pytest.mark.parametrize("task", _SUPPORTED_TASKS)
def test_inference_async_dynamic(task):
    executor = ThreadPoolExecutor(max_workers =2)
    pipeline = Pipeline.create(task=task, batch_size=None, executor=executor)
    inputs = create_test_inputs(task=task, batch_size=5)
    outputs = pipeline(**inputs)
    _test_inference_timing(outputs.inference_timing)

def test_inference_sync():
    pipeline = FakePipeline("")
    outputs = pipeline(sentence="all your base are belong to us")
    _test_inference_timing(outputs.inference_timing)

def test_inference_async():
    pipeline = FakePipeline("", batch_size=1, executor=ThreadPoolExecutor(max_workers=2))
    outputs = pipeline(sentence="all your base are belong to us")
    _test_inference_timing(outputs.inference_timing)
#
# def test_inference_async_dynamic(self, setup):
#     model, stub, task, executor = setup
#     pipeline = Pipeline.create(task="image_classification", batch_size=None, executor=executor)
#     outputs = pipeline()
#
#     self._test_inference_timing(task, pipeline, pipeline_input)


def _test_inference_timing(inference_timing):
    for name, value in inference_timing:
        assert isinstance(value, float)
    assert (
        inference_timing.engine_forward_delta > inference_timing.pre_process_delta
    )
    assert (
        inference_timing.engine_forward_delta > inference_timing.post_process_delta
    )
    assert (
        inference_timing.total_inference_delta
        == inference_timing.pre_process_delta
        + inference_timing.engine_forward_delta
        + inference_timing.post_process_delta
    )
