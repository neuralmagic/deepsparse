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


import json
import os
import shutil
import tempfile

import numpy

import pytest
from deepsparse import Pipeline
from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from sparsezoo import Model
from sparsezoo.utils import load_numpy_list


from PIL import Image  # isort:skip
from torchvision import transforms  # isort:skip


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
class TestImageClassification:
    @pytest.fixture()
    def setup(self, zoo_stub, image_size, num_samples):

        yield zoo_stub, image_size, num_samples

    def test_image_classification_pipeline_preprocessing(self, setup):
        zoo_stub, image_size, num_samples = setup

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
        zoo_model = Model(zoo_stub)
        data_originals_path = None
        if zoo_model.sample_originals is not None:
            if not zoo_model.sample_originals.files:
                zoo_model.sample_originals.unzip()
            data_originals_path = zoo_model.sample_originals.path
        for idx, sample in enumerate(load_numpy_list(data_originals_path)):
            image_raw = list(sample.values())[0]
            image_raw = Image.fromarray(image_raw)

            preprocessed_image_pipeline = ic_pipeline.process_inputs(
                ic_pipeline.input_schema(images=[image_raw])
            )[0][0]
            preprocessed_image_standard = standard_imagenet_transforms(
                image_raw
            ).numpy()

            abs_max_diff = numpy.max(
                numpy.abs(preprocessed_image_pipeline - preprocessed_image_standard)
            )
            assert abs_max_diff < 1e-5

            if idx + 1 >= num_samples:
                break

    def test_image_classification_with_config(self, setup):
        zoo_stub, image_size, _ = setup

        expected_class_name = "dummy_class_name"

        dirpath = tempfile.mkdtemp()
        zoo_model = Model(zoo_stub, dirpath)
        deployment_path = zoo_model.deployment.default.path

        # 1000 labels because the stub is a model trained on imagenet
        labels_to_class_mapping = {label: expected_class_name for label in range(1000)}
        config = {"labels_to_class_mapping": labels_to_class_mapping}

        # create config.json in deployment directory
        with open(os.path.join(deployment_path, "config.json"), "w") as fp:
            json.dump(config, fp)

        pipeline = Pipeline.create("image_classification", zoo_model.path)
        output = pipeline(images=[numpy.random.rand(3, image_size, image_size)])

        assert all([label == expected_class_name for label in output.classes])
        shutil.rmtree(dirpath)
