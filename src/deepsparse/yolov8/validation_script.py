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

"""
Usage: deepsparse.image_classification.eval [OPTIONS]

  Validation Script for Image Classification Models

Options:
  --dataset-path, --dataset_path DIRECTORY
                                  Path to the validation dataset  [required]
  --model-path, --model_path TEXT
                                  Path/SparseZoo stub for the Image
                                  Classification model to be evaluated.
                                  Defaults to dense (vanilla) resnet50 trained
                                  on Imagenette  [default: zoo:cv/classificati
                                  on/resnet_v1-50/pytorch/sparseml/imagenette/
                                  base-none]
  --batch-size, --batch_size INTEGER
                                  Test batch size, must divide the dataset
                                  evenly, else last batch will be dropped
                                  [default: 1]
  --image-size, --image_size INTEGER
                                  integer size to evaluate images at (will be
                                  reshaped to square shape)  [default: 224]
  --num-cores, --num_cores INTEGER
                                  Number of CPU cores to run deepsparse with,
                                  default is all available
  --dataset-kwargs, --dataset_kwargs TEXT
                                  Keyword arguments to be passed to dataset
                                  constructor, should be specified as a json
  --engine [deepsparse|onnxruntime]
                                  engine type to use, valid choices:
                                  ['deepsparse', 'onnxruntime']  [default:
                                  deepsparse]
  --help                          Show this message and exit.

#########
EXAMPLES
#########

##########
Example command for validating pruned resnet50 on imagenette dataset:
python validation_script.py \
  --dataset-path /path/to/imagenette/

"""
import json
from typing import Dict

import click
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from deepsparse.pipeline import Pipeline
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

resnet50_imagenet_pruned = (
    "yolov8n.onnx"
)


def parse_json_callback(ctx, params, value: str) -> Dict:
    """
    Parse a json string into a dictionary
    :param ctx: The click context
    :param params: The click params
    :param value: The json string to parse
    :return: The parsed dictionary
    """
    # JSON string -> dict Callback
    if isinstance(value, str):
        return json.loads(value)
    return value


@click.command()
@click.option(
    "--dataset-path",
    "--dataset_path",
    required=False,
    type=click.Path(dir_okay=True, file_okay=False),
    help="Path to the validation dataset",
)
@click.option(
    "--model-path",
    "--model_path",
    type=str,
    default=resnet50_imagenet_pruned,
    help="Path/SparseZoo stub for the Image Classification model to be "
    "evaluated. Defaults to dense (vanilla) resnet50 trained on Imagenette",
    show_default=True,
)
@click.option(
    "--batch-size",
    "--batch_size",
    type=int,
    default=1,
    show_default=True,
    help="Test batch size, must divide the dataset evenly, else last "
    "batch will be dropped",
)
@click.option(
    "--image-size",
    "--image_size",
    type=int,
    default=640,
    show_default=True,
    help="integer size to evaluate images at (will be reshaped to square shape)",
)
@click.option(
    "--num-cores",
    "--num_cores",
    type=int,
    default=None,
    show_default=True,
    help="Number of CPU cores to run deepsparse with, default is all available",
)
@click.option(
    "--dataset-kwargs",
    "--dataset_kwargs",
    default=json.dumps({}),
    type=str,
    callback=parse_json_callback,
    help="Keyword arguments to be passed to dataset constructor, "
    "should be specified as a json object",
)
@click.option(
    "--engine",
    default=DEEPSPARSE_ENGINE,
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE]),
    show_default=True,
    help="engine type to use, valid choices: ['deepsparse', 'onnxruntime']",
)
def main(
    dataset_path: str,
    model_path: str,
    batch_size: int,
    image_size: int,
    num_cores: int,
    dataset_kwargs: Dict,
    engine: str,
):
    """
    Validation Script for Image Classification Models
    """

    if "resize_scale" in dataset_kwargs:
        resize_scale = dataset_kwargs["resize_scale"]
    else:
        resize_scale = 256.0 / 224.0  # standard used

    if "resize_mode" in dataset_kwargs:
        resize_mode = dataset_kwargs["resize_mode"]
    else:
        resize_mode = "bilinear"

    if "rgb_means" in dataset_kwargs:
        rgb_means = dataset_kwargs["rgb_means"]
    else:
        rgb_means = IMAGENET_RGB_MEANS

    if "rgb_stds" in dataset_kwargs:
        rgb_stds = dataset_kwargs["rgb_stds"]
    else:
        rgb_stds = IMAGENET_RGB_STDS

    if type(resize_mode) is str and resize_mode.lower() in ["linear", "bilinear"]:
        interpolation = transforms.InterpolationMode.BILINEAR
    elif type(resize_mode) is str and resize_mode.lower() in ["cubic", "bicubic"]:
        interpolation = transforms.InterpolationMode.BICUBIC

    dataset = torchvision.datasets.CocoDetection(
        root="./coco/val2017",
        annFile= "./coco/annotations/instances_val2017.json",
        transform=transforms.Compose([
                transforms.ToTensor(),
            ]
        ),
    )

    data_loader = dataset

    pipeline = Pipeline.create(
        task="yolov8",
        model_path=model_path,
        batch_size=batch_size,
        num_cores=num_cores,
        engine_type=engine,
    )
    print(f"engine info: {pipeline.engine}")
    correct = total = 0
    progress_bar = tqdm(data_loader)
    metric = MeanAveragePrecision()
    targets = []
    predictions = []

    for u, batch in enumerate(progress_bar):
        batch, annotations = batch
        batch = batch.numpy()
        outs = pipeline(images=batch * 255)

        predicted_labels = outs.labels[0]
        predicted_labels = [int(float(x)) +1 for x in predicted_labels]
        predicted_bboxes = outs.boxes[0]
        predicted_scores = outs.scores[0]

        actual_labels = [ann['category_id'] for ann in annotations]
        actual_bboxes = [ann['bbox'] for ann in annotations]

        import matplotlib.pyplot as plt
        import cv2
        import numpy as np

        batch = 255 * batch
        batch = batch.astype(np.uint8).transpose(1, 2, 0).copy()
        h, w, _ = batch.shape
        _pred_boxes = []
        for bbox in predicted_bboxes:
            bbox = np.array(bbox)
            scale = np.flipud(np.divide(
                    np.asarray((h,w)), np.asarray((640,640))
                )
            )
            scale = np.concatenate([scale, scale])
            bbox = np.multiply(bbox, scale)
            bbox = bbox.tolist()
            cv2.rectangle(batch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            _pred_boxes.append([bbox[0], bbox[2], bbox[1], bbox[3]])

        # _act_bbox = []
        # for bbox in actual_bboxes:
        #     x_min, y_min, width, height = bbox
        #     # get (x,y) and (x2,y2) coordinates
        #     x_max = x_min + width
        #     y_max = y_min + height
        #     cv2.rectangle(batch, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        #     _act_bbox.append([x_min, y_min, x_max, y_max])
        #
        plt.imshow(batch /255)
        plt.show()
        # from torchvision.ops import box_iou
        # iou = box_iou(torch.tensor(_pred_boxes).reshape(-1,4), torch.tensor(_act_bbox).reshape(-1,4))
        pass
    #             print(iou)
    #             if iou > 0.5:
    #
    #                 predictions.append(dict(boxes=torch.tensor([_pred_boxes[i]]),
    #                             scores = torch.tensor([predicted_scores[i]]),
    #                             labels = torch.tensor([predicted_labels[i]])))
    #
    #                 targets.append(dict(boxes=torch.tensor([_act_bbox[j]]),
    #                                     labels=torch.tensor([actual_labels[j]])))
    #                 print(predictions, targets)
    #
    #     if u == 100:
    #         break
    # metric.update(predictions, targets)
    # print(metric.compute())







if __name__ == "__main__":
    main()
