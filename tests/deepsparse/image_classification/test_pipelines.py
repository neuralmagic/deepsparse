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
from PIL import Image
from torchvision import transforms

import pytest
from deepsparse import Pipeline
from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from sparsezoo import Zoo
from sparsezoo.utils import load_numpy_list


@pytest.mark.parametrize(
    "zoo_stub,image_size,num_samples",
    [
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned85_quant-none-vnni",
            224,
            5,
        )
    ],
)
def test_image_classification_pipeline_preprocessing(zoo_stub, image_size, num_samples):
    non_rand_resize_scale = 256.0 / 224.0  # standard used
    standard_imagenet_transforms = transforms.Compose(
        [
            transforms.Resize(round(non_rand_resize_scale * image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
        ]
    )

    ic_pipeline = Pipeline.create("image_classification", zoo_stub)
    zoo_model = Zoo.load_model_from_stub(zoo_stub)
    data_originals_path = zoo_model.data_originals.downloaded_path()

    for idx, sample in enumerate(load_numpy_list(data_originals_path)):
        image_raw = list(sample.values())[0]
        image_raw = Image.fromarray(image_raw)

        preprocessed_image_pipeline = ic_pipeline.process_inputs(
            ic_pipeline.input_schema(images=[image_raw])
        )[0][0]
        preprocessed_image_standard = standard_imagenet_transforms(image_raw).numpy()

        abs_max_diff = numpy.max(
            numpy.abs(preprocessed_image_pipeline - preprocessed_image_standard)
        )
        assert abs_max_diff < 1e-5

        if idx + 1 >= num_samples:
            break
