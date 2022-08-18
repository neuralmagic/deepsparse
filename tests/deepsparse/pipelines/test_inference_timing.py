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


@pytest.mark.parametrize(
    "task, stub",
    [
        (
            "question_answering",
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned95_obs_quant-none",  # noqa E501
        ),
        (
            "image_classification",
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95_quant-none",  # noqa E501
        ),
        (
            "yolact",
            "zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/pruned82_quant-none",  # noqa E501
        ),
        (
            "yolo",
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",  # noqa E501
        ),
    ],
)
class TestInferenceTiming:
    @pytest.fixture
    def setup(self, task, stub):
        download_dir = tempfile.mktemp()
        pipeline = Pipeline.create(task, stub)

        sample_input = Model(stub, download_dir).sample_batch(1)["sample_inputs"]
        pipeline_input = [arr[0] for arr in sample_input]

        yield pipeline_input, pipeline, task

        shutil.rmtree(download_dir)

    def test_inference_correctness(self, setup):
        pipeline_input, pipeline, task = setup
        if task in ["question_answering"]:
            pipeline(question="What's my name?", context="My name is Snorlax")
        else:
            pipeline(images=pipeline_input)

        inference_timing = pipeline.inference_timing
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
