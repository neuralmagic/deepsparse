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

import shutil
import tempfile

import pytest
from deepsparse import Pipeline
from sparsezoo import Model
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.parametrize(
    "task, stub",
    [
        # (
        #     "question_answering",
        #     "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
        # ),
        (
            "image_classification",
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none",  # noqa E501
        ),
        # (
        #     "yolact",
        #     "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none",  # noqa E501
        # ),
        # (
        #     "yolo",
        #     "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
        # ),
    ],
    scope = "class"
)
class TestInferenceTiming:
    @pytest.fixture()
    def setup(self, task, stub):
        download_dir = tempfile.mktemp()

        sample_input = Model(stub, download_dir).sample_batch(5)["sample_inputs"]
        pipeline_input = sample_input #[arr[0] for arr in sample_input]

        yield pipeline_input, stub, task

        shutil.rmtree(download_dir)

    def test_inference_sync(self, setup):
        pipeline_input, stub, task = setup
        pipeline = Pipeline.create(task, stub)

        if task in ["question_answering"]:
            _, inference_timing = pipeline(question="What's my name?", context="My name is Snorlax")
        else:
            _, inference_timing = pipeline(images=pipeline_input)

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

    def test_inference_async(self, setup):
        pipeline_input, stub, task = setup

        executor = ThreadPoolExecutor()
        pipeline = Pipeline.create(task, stub, executor = executor)

        if task in ["question_answering"]:
            _, inference_timing = pipeline(question="What's my name?", context="My name is Snorlax").result()
        else:
            _, inference_timing = pipeline(images=pipeline_input).result()

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

    def test_inference_async_dynamic(self, setup):
        pipeline_input, stub, task = setup

        executor = ThreadPoolExecutor()
        pipeline = Pipeline.create(task=task, batch_size=None, executor=executor)

        if task in ["question_answering"]:
            _, inference_timing = pipeline(question="What's my name?", context="My name is Snorlax")
        else:
            _, inference_timing = pipeline(images=pipeline_input).result()

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

