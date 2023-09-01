# neuralmagic: no copyright
# flake8: noqa

import os

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
        args=None,
    ):
        DetectionValidator.__init__(self, dataloader, save_dir, pbar, args)
        DeepSparseValidator.__init__(self, pipeline)

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
