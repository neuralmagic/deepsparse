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

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

import torch
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    SETTINGS,
    TQDM_BAR_FORMAT,
    callbacks,
)
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode


__all__ = ["DeepSparseValidator"]


def schema_to_tensor(pipeline_outputs: YOLOOutput, device: str) -> List[torch.Tensor]:
    """
    Transform the YOLOOutput to the format expected by the validation code.

    :param pipeline_outputs: YOLOOutput from the pipeline
    :param device: device to move the tensors to
    :return list of tensor with the format [x1, y1, x2, y2, confidence, class]
    """

    preds = []

    for boxes, labels, confidence in zip(
        pipeline_outputs.boxes, pipeline_outputs.labels, pipeline_outputs.scores
    ):

        boxes = torch.tensor(boxes)

        # map labels to integers and reshape for concatenation
        labels = list(map(int, list(map(float, labels))))
        labels = torch.tensor(labels).view(-1, 1)

        # reshape for concatenation
        scores = torch.tensor(confidence).view(-1, 1)
        # concatenate and append to preds
        preds.append(torch.cat([boxes, scores, labels], axis=1).to(device))
    return preds


# adapted from ULTRALYTICS GITHUB:
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/validator.py
# the appropriate edits are marked with # deepsparse edit: <edit comment>


class DeepSparseValidator(BaseValidator):  # deepsparse edit: overwriting BaseValidator
    """
    A DeepSparseValidator class for creating validators for
    YOLOv8 Deepsparse pipeline.

    Attributes:
        pipeline (Pipeline): DeepSparse Pipeline to be evaluated
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        logger (logging.Logger): Logger to use for validation.
        args (SimpleNamespace): Configuration for the validator.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        speed (float): Batch processing speed in seconds.
        jdict (dict): Dictionary to store validation results.
        save_dir (Path): Directory to save results.
    """

    def __init__(
        self,
        pipeline: Pipeline,  # deepsparse edit: added pipeline
        dataloader=None,
        save_dir=None,
        pbar=None,
        logger=None,
        args=None,
    ):
        """
        Initializes a DeepSparseValidator instance.
        Args:
            pipeline (Pipeline): DeepSparse Pipeline to be evaluated
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            logger (logging.Logger): Logger to log messages.
            args (SimpleNamespace): Configuration for the validator.
        """
        self.pipeline = pipeline  # deepsparse edit: added pipeline
        self.dataloader = dataloader
        self.pbar = pbar
        self.logger = logger or LOGGER
        self.args = args or get_cfg(DEFAULT_CFG)
        self.model = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = False
        self.speed = None
        self.jdict = None

        project = self.args.project or Path(SETTINGS["runs_dir"]) / self.args.task
        name = self.args.name or f"{self.args.mode}"
        self.save_dir = save_dir or increment_path(
            Path(project) / name,
            exist_ok=self.args.exist_ok if RANK in {-1, 0} else True,
        )
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(
            parents=True, exist_ok=True
        )

        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001

        self.callbacks = defaultdict(list, callbacks.default_callbacks)  # add callbacks

    @smart_inference_mode()
    # deepsparse edit: replaced arguments `trainer` and `model`
    # with `stride` and `classes`
    def __call__(self, stride: int, classes: Dict[int, str]):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        # deepsparse edit: removed the if-statement responsible
        # for validation when self.training is True
        callbacks.add_integration_callbacks(self)
        self.run_callbacks("on_val_start")
        self.device = select_device(self.args.device, self.args.batch)
        self.args.half &= self.device.type != "cpu"
        self.data = check_det_dataset(self.args.data)
        if self.device.type == "cpu":
            self.args.workers = (
                0  # faster CPU val as time dominated by inference, not dataloading
            )

        self.dataloader = self.dataloader or self.get_dataloader(
            self.data.get("val") or self.data.set("test"), self.args.batch
        )
        # deepsparse edit: left only profiler for inference, removed the redundant
        # profilers for pre-process, loss and post-process
        dt = Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        # deepsparse edit: replaced argument `model` with `classes`
        self.init_metrics(classes=classes)
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            # deepsparse edit:
            # - removed the redundant pre-process function
            # - removed the redundant loss computation
            # - removed the redundant post-process function

            # deepsparse edit: replaced the inference model with the DeepSparse pipeline
            # inference
            with dt:
                outputs = self.pipeline(
                    images=[x.cpu().numpy() for x in batch["img"]],
                    iou_thres=self.args.iou,
                    conf_thres=self.args.conf,
                    multi_label=True,
                )
            preds = schema_to_tensor(pipeline_outputs=outputs, device=self.device)
            batch["bboxes"] = batch["bboxes"].to(self.device)
            batch["cls"] = batch["cls"].to(self.device)
            batch["batch_idx"] = batch["batch_idx"].to(self.device)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = dt.t / len(self.dataloader.dataset) * 1e3  # speeds per image
        self.run_callbacks("on_val_end")

        # deepsparse_edit: changed the string formatting to match the
        # removed profilers
        self.logger.info("Speed: %.1fms inference per image" % self.speed)

        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                self.logger.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        return stats
