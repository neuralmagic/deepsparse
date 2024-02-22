# neuralmagic: no copyright
# flake8: noqa

import os

from deepsparse import Pipeline
from deepsparse.yolov8.utils.validation.deepsparse_validator import DeepSparseValidator
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix
from ultralytics.yolo.v8.segment import SegmentationValidator


__all__ = ["DeepSparseSegmentationValidator"]


# adapted from ULTRALYTICS GITHUB:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/v8/segment/val.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>
class DeepSparseSegmentationValidator(DeepSparseValidator, SegmentationValidator):
    def __init__(
        self,
        pipeline: Pipeline,
        dataloader=None,
        save_dir=None,
        pbar=None,
        logger=None,
        args=None,
    ):
        SegmentationValidator.__init__(self, dataloader, save_dir, pbar, args)
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
        # deepsparse edit: set number of mask protos to 32
        self.nm = 32
        self.names = classes
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.plot_masks = []
        self.seen = 0
        self.jdict = []
        self.stats = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster
