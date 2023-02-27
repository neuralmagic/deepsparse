# Ultralytics YOLO 🚀, GPL-3.0 license

import os

from ultralytics.yolo.utils import NUM_THREADS, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, SegmentMetrics, box_iou, mask_iou
from typing import Dict
from ultralytics.yolo.v8.segment import SegmentationValidator
import json
from tqdm import tqdm
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import TQDM_BAR_FORMAT, callbacks, emojis
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode

from deepsparse import Pipeline
from deepsparse.yolov8.utils.validation.helpers import schema_to_tensor

__all__ = ["DeepSparseSegmentationValidator"]


class DeepSparseSegmentationValidator(SegmentationValidator):

    # deepsparse edit: add pipeline to init
    def __init__(self,
                 pipeline: Pipeline,
                 dataloader=None,
                 save_dir=None,
                 pbar=None,
                 logger=None,
                 args=None):
        super().__init__(dataloader, save_dir, pbar, logger, args)
        self.training = False
        self.pipeline = pipeline

    # deepsparse edit: replaced argument `model` with `classes`
    def init_metrics(self, classes):
        val = self.data.get('val', '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
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
            check_requirements('pycocotools>=2.0.6')
            self.process = ops.process_mask_upsample  # more accurate
        else:
            self.process = ops.process_mask  # faster

    @smart_inference_mode()
    # deepsparse edit: replaced arguments `trainer` and `model`
    # with `stride` and `classes`
    def __call__(self, classes: Dict[int, str]):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        # deepsparse edit: removed the if-statement responsible
        # for validation when self.training is True
        callbacks.add_integration_callbacks(self)
        self.run_callbacks('on_val_start')
        self.device = select_device(self.args.device, self.args.batch)
        self.args.half &= self.device.type != 'cpu'
        self.data = check_det_dataset(self.args.data)
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading

        self.dataloader = self.dataloader or \
                          self.get_dataloader(self.data.get("val") or self.data.set("test"), self.args.batch)


        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(classes=classes)
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # pre-process
            with dt[0]:
                batch = self.preprocess(batch)

            # inference
            with dt[1]:
                outputs = self.pipeline(images=[x.cpu().numpy() * 255 for x in batch["img"]],
                iou_thres = self.args.iou,
                conf_thres = self.args.conf,
                multi_label = True,
                return_intermediate_outputs = True
            )
            preds = schema_to_tensor(pipeline_outputs=outputs, device=self.device)

            # pre-process predictions
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = tuple(x.t / len(self.dataloader.dataset) * 1E3 for x in dt)  # speeds per image
        self.run_callbacks('on_val_end')

        self.logger.info('Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms post-process per image' %
                         self.speed)
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), 'w') as f:
                self.logger.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        return stats
