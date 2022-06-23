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

import cv2
from deepsparse import Pipeline
from deepsparse.yolact.utils.annotate import annotate_image


src = "golfish.jpeg"
cv_pipeline = Pipeline.create(
    task="yolact",
    model_path="zoo:cv/segmentation/yolact-darknet53/pytorch/dbolya/coco/base-none",
    batch_size=2,
)
img_numpy = cv2.imread(src)
img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)

inference = cv_pipeline(images=[src])
annotate_image(img_numpy, inference)
