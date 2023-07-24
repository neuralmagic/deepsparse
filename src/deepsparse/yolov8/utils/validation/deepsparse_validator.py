# neuralmagic: no copyright
# flake8: noqa

import json
from pathlib import Path
from typing import Dict

from tqdm import tqdm

import torch
from deepsparse.yolov8.utils.validation.helpers import schema_to_tensor
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.utils import LOGGER, TQDM_BAR_FORMAT, callbacks
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode


__all__ = ["DeepSparseValidator"]


class DeepSparseValidator:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.training = False

    @smart_inference_mode()
    # deepsparse edit: replaced arguments `trainer` and `model`
    # `classes`
    def __call__(self, classes: Dict[int, str]):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        # deepsparse edit: removed the if-statement responsible
        # for validation when self.training is True
        callbacks.add_integration_callbacks(self)
        self.run_callbacks("on_val_start")
        if not torch.cuda.is_available():
            self.args.device = "cpu"
        self.device = select_device(self.args.device, self.args.batch)
        self.data = check_det_dataset(self.args.data)
        if isinstance(self.data["path"], str):
            self.data["path"] = Path(self.data["path"])

        if self.device.type == "cpu":
            self.args.workers = (
                0  # faster CPU val as time dominated by inference, not dataloading
            )

        self.dataloader = self.dataloader or self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

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
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # pre-process
            with dt[0]:
                batch = self.preprocess(batch)

            # inference
            with dt[1]:
                outputs = self.pipeline(
                    images=[x.cpu().numpy() * 255 for x in batch["img"]],
                    return_intermediate_outputs=True,
                )
            preds = schema_to_tensor(pipeline_outputs=outputs, device=self.device)

            # pre-process predictions
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )  # speeds per image
        self.finalize_metrics()
        self.run_callbacks("on_val_end")

        LOGGER.info(
            "Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms post-process per image"
            % tuple(self.speed.values())
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        return stats
