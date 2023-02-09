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

# flake8: noqa

import os

import torch
from deepsparse import Pipeline
from deepsparse.yolov8.utils.validation.deepsparse_validator import DeepSparseValidator
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics
from ultralytics.yolo.v8.detect.val import DetectionValidator


__all__ = ["DeepSparseDetectionValidator"]

# adapted from ULTRALYTICS GITHUB:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/v8/detect/val.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
class DeepSparseDetectionValidator(DeepSparseValidator, DetectionValidator):
    def __init__(
        self,
        pipeline: Pipeline,
        dataloader=None,
        save_dir=None,
        pbar=None,
        logger=None,
        args=None,
    ):
        super().__init__(pipeline, dataloader, save_dir, pbar, logger, args)
        self.args.task = "detect"
        self.is_coco = False
        self.class_map = None
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    # deepsparse edit: replaced argument `model` with `classes`
    def init_metrics(self, classes):
        val = self.data.get("val", "")  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(
            f"coco{os.sep}val2017.txt"
        )  # is COCO dataset
        self.class_map = (
            ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        )
        self.args.save_json |= (
            self.is_coco and not self.training
        )  # run on final val if training COCO
        self.nc = len(classes)
        self.names = classes
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.seen = 0
        self.jdict = []
        self.stats = []
