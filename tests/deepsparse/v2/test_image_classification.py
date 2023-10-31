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
from deepsparse.v2.image_classification import ImageClassificationPipeline
from deepsparse.v2.image_classification.preprocess_operator import (
    ImageClassificationInput,
)
from tests.deepsparse.pipelines.data_helpers import computer_vision


@pytest.fixture
def get_images():
    batch_size = 2
    images = computer_vision(batch_size=batch_size)
    return images.get("images")


def test_image_classification(get_images):
    model_path = (
        "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned95-none"
    )
    pipeline = ImageClassificationPipeline(model_path=model_path)
    output = pipeline(ImageClassificationInput(images=get_images))
    assert output.labels == [[207], [670]]
    assert numpy.allclose(output.scores, [[21.85], [17.33]], atol=0.01)
