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

import pytest
from deepsparse.pipelines.computer_vision import ComputerVisionSchema
from tests.deepsparse.pipelines.data_helpers import computer_vision


def _get_images():
    batch_size = 5
    images = computer_vision(batch_size=batch_size)
    return images.get("images")


def test_accepts_input():
    ComputerVisionSchema(images="asdf")
    ComputerVisionSchema(images=["asdf", "qwer"])
    ComputerVisionSchema(images=numpy.zeros((1, 2, 3)))
    ComputerVisionSchema(images=[numpy.zeros((1, 2, 3)), numpy.zeros((1, 2, 3))])


@pytest.mark.parametrize(
    "schema_cls, image_files", [(ComputerVisionSchema, _get_images())]
)
def test_from_files(schema_cls, image_files):
    image_iters = (open(image, "rb") for image in image_files)

    expected = schema_cls(
        images=[numpy.array(Image.open(image)) for image in image_files]
    )
    actual = schema_cls.from_files(files=image_iters)

    assert isinstance(actual, schema_cls)
    assert len(actual.images) == len(expected.images)

    for actual_img, expected_img in zip(actual.images, expected.images):
        assert actual_img.shape == expected_img.shape
