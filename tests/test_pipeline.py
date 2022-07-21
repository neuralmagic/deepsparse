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

from typing import OrderedDict

import numpy
from PIL import Image
from torchvision import transforms

import pytest
from deepsparse.image_classification import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
    ImageClassificationInput,
    ImageClassificationOutput,
)
from deepsparse.pipeline import Pipeline
from sparsezoo import Zoo
from sparsezoo.utils import load_numpy_list


@pytest.mark.parametrize(
    "zoo_stub,image_size",
    [
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned85_quant-none-vnni",
            224,
        )
    ],
)
def test_function_pipeline_as_image_classifier(zoo_stub, image_size):
    non_rand_resize_scale = 256.0 / 224.0  # standard used
    standard_imagenet_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(round(non_rand_resize_scale * image_size)),
            transforms.CenterCrop(image_size),
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
        ]
    )

    def preprocess(input: ImageClassificationInput):
        assert len(input.images) == 1
        assert isinstance(input, ImageClassificationInput)
        assert isinstance(input.images, list)
        assert all(isinstance(i, numpy.ndarray) for i in input.images)
        return [
            standard_imagenet_transforms(Image.fromarray(img)).unsqueeze(0).numpy()
            for img in input.images
        ]

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
        zoo_stub,
        input_schema=ImageClassificationInput,
        output_schema=ImageClassificationOutput,
        process_inputs_fn=preprocess,
        process_outputs_fn=postprocess,
    )

    zoo_model = Zoo.load_model_from_stub(zoo_stub)
    data_originals_path = zoo_model.data_originals.downloaded_path()
    sample = load_numpy_list(data_originals_path)[0]
    assert isinstance(sample, OrderedDict)
    assert len(sample) == 1
    image_raw = list(sample.values())[0]
    assert isinstance(image_raw, numpy.ndarray)
    input = ImageClassificationInput(images=[image_raw])
    output = pipeline(input)
    assert isinstance(output, ImageClassificationOutput)
