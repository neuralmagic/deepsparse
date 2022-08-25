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

import numpy

import pytest
from deepsparse.image_classification import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    ImageClassificationOutput,
)
from deepsparse.pipelines.computer_vision import ComputerVisionSchema
from deepsparse.pipeline import Pipeline
from deepsparse.pipelines.custom_pipeline import CustomTaskPipeline
from deepsparse.utils.onnx import model_to_path


# NOTE: these need to be placed after the other imports bc of a dependency chain issue
from PIL import Image  # isort:skip
from torchvision import transforms  # isort:skip


@pytest.fixture(scope="module")
def model_path():
    stub = (
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml"
        "/imagenet/pruned85_quant-none-vnni"
    )
    yield model_to_path(stub)


@pytest.mark.parametrize(
    "task_name",
    [
        "custom",
        "customtask",
        "custom_task",
        "custom-task",
        "custom-image-classification",
    ],
)
def test_custom_pipeline_task_names(task_name):
    cls = Pipeline._get_task_constructor(task_name)
    assert cls == CustomTaskPipeline


def test_no_input_call(model_path):
    pipeline = Pipeline.create(task="custom", model_path=model_path)
    assert isinstance(pipeline, CustomTaskPipeline)

    assert pipeline.input_schema == object
    assert pipeline.output_schema == object
    assert pipeline.process_inputs(1.2345) == 1.2345
    assert pipeline.process_engine_outputs([1.2345], asdf=True) == [1.2345]

    image_raw = numpy.random.random(size=(1, 3, 224, 224)).astype(numpy.float32)

    # actually run the pipeline
    output = pipeline([image_raw])
    assert isinstance(output, list)
    # NOTE: image classifier outputs 2 arrays per input (logits & softmax)
    assert len(output) == 2


def test_custom_pipeline_as_image_classifier(model_path):
    image_size = 224
    non_rand_resize_scale = 256.0 / 224.0  # standard used
    standard_imagenet_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(round(non_rand_resize_scale * image_size)),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
        ]
    )

    def preprocess(input: List[ComputerVisionSchema]):
        # assert len(input.images) == 1
        # assert isinstance(input, list)
        # assert isinstance(input.images, list)
        # assert all(isinstance(i, numpy.ndarray) for i in input.images)
        # return [
        #     standard_imagenet_transforms(Image.fromarray(img)).unsqueeze(0).numpy()
        #     for img in input.images
        # ]

    def postprocess(outputs, **kwargs):
        assert len(outputs) == 2  # NOTE: logits & softmax for this model
        labels, label_scores = [], []
        for scores in outputs[1]:
            lbl = scores.argmax(-1)
            labels.append(lbl)
            label_scores.append(scores[lbl])

        return ImageClassificationOutput(
            scores=label_scores,
            labels=labels,
        )

    pipeline = Pipeline.create(
        "custom",
        model_path,
        input_schema=ComputerVisionSchema,
        output_schema=ImageClassificationOutput,
        process_inputs_fn=preprocess,
        process_outputs_fn=postprocess,
    )
    assert isinstance(pipeline, CustomTaskPipeline)

    # load model & data
    image_raw = (numpy.random.random(size=(224, 224, 3)) * 255).astype(numpy.uint8)

    # actually run the pipeline
    output = pipeline(image_raw)
    assert isinstance(output, ImageClassificationOutput)
