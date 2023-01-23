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

from typing import Callable, List

import torch
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput, YOLOPipeline
from ultralytics.yolo.utils import ops


@Pipeline.register(
    task="yolov8",
    default_model_path=None,
)
class YOLOv8Pipeline(YOLOPipeline):
    def __init__(
        self, nms_function: Callable = ops.non_max_suppression, **kwargs
    ):  # TODO: finish typing here
        self.nms_function = nms_function
        super().__init__(nms_function=nms_function, **kwargs)
