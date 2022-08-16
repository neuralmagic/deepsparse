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

import pytest
from deepsparse import Pipeline
from sparsezoo import Model


@pytest.mark.parametrize(
    "zoo_stub,image_size",
    [
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96",
            640,
        )
    ],
)
@pytest.mark.smoke
class TestYOLO:
    @pytest.fixture()
    def setup(self, zoo_stub):

        yield zoo_stub

    def test_yolact_with_config(self, setup):
        zoo_stub, _ = setup

        expected_class_name = "dummy_class_name"

        dirpath = tempfile.mkdtemp()
        zoo_model = Model(zoo_stub, dirpath)
        deployment_path = zoo_model.deployment.default.path
        input_batch = zoo_model.sample_batch(1)["sample_inputs"]

        # 80 labels because the stub is a model trained on coco
        labels_to_class_mapping = {label: expected_class_name for label in range(80)}
        config = {"labels_to_class_mapping": labels_to_class_mapping}

        # create config.json in deployment directory
        with open(os.path.join(deployment_path, "config.json"), "w") as fp:
            json.dump(config, fp)

        pipeline = Pipeline.create("yolo", zoo_model.path)
        output = pipeline(images=input_batch)

        assert all([label == expected_class_name for label in output.classes[0]])
        shutil.rmtree(dirpath)
