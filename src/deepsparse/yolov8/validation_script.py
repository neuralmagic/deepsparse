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
from torchvision.ops import complete_box_iou, box_convert
from deepsparse.image_classification.constants import (
    IMAGENET_RGB_MEANS,
    IMAGENET_RGB_STDS,
)
from deepsparse.pipeline import Pipeline
import matplotlib.pyplot as plt
import cv2
import numpy as np
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
    default=ORT_ENGINE,
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
    from typing import Tuple, Any
    if type(resize_mode) is str and resize_mode.lower() in ["linear", "bilinear"]:
        interpolation = transforms.InterpolationMode.BILINEAR
    elif type(resize_mode) is str and resize_mode.lower() in ["cubic", "bicubic"]:
        interpolation = transforms.InterpolationMode.BICUBIC


    class Test(torchvision.datasets.CocoDetection):
        def __init__(self, root, annFile, transform=None):
            super().__init__(root, annFile, transform=transform)

        def __getitem__(self, index: int) -> Tuple[Any, Any]:
            id = self.ids[index]
            image = self._load_image(id)
            target = self._load_target(id)

            if self.transforms is not None:
                image, target = self.transforms(image, target)

            return image, target, self.coco.imgs[self.ids[index]]


    dataset = Test(
        root="./coco/val2017",
        annFile= "./coco/annotations/instances_val2017.json",
        transform=transforms.Compose([
                transforms.ToTensor(),
            ]
        ),
    )

    data_loader = dataset

    pipeline = Pipeline.create(
        task="yolo",
        model_path="zoo:cv/detection/yolo_v3-spp/pytorch/ultralytics/coco/pruned_quant-aggressive_94",
        batch_size=batch_size,
        num_cores=num_cores,
        engine_type=engine,
    )
    print(f"engine info: {pipeline.engine}")
    import glob
    image_paths = glob.glob("./coco/val2017/*")
    results = []

    for u, batch in enumerate(tqdm(data_loader)):
        outs = pipeline(images="./coco/val2017/" + batch[2]['file_name'])
        predicted_labels = [int(float(x)) +1 for x in outs.labels[0]]
        predicted_bboxes = outs.boxes[0]
        predicted_scores = outs.scores[0]

        imageId = batch[2]['id']

        ## scaling bboxes ##
        predicted_bboxes = np.array(predicted_bboxes)
        _, h, w = batch[0].shape
        scale = np.flipud(np.divide(np.asarray((h, w)), np.asarray((640, 640))))
        scale = np.concatenate([scale, scale])
        if not predicted_bboxes.any():
            continue
        predicted_bboxes = np.multiply(predicted_bboxes, scale).tolist()

        img = batch[0].numpy().transpose(1, 2, 0).copy()
        img = 255 * img
        img= img.astype(np.uint8)

        for label, bbox, conf in zip(predicted_labels, predicted_bboxes, predicted_scores):
            x_min, y_min, x_max, y_max = bbox
            w = x_max - x_min
            h = y_max - y_min

            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)


            results.append({"image_id": imageId,
                            "category_id": label,
                            "bbox": [x_min, y_min, h,w],
                            "score": conf})
            plt.imshow(img)
            plt.show()
            pass




    with open("test.json", "w") as f:
        json.dump(results, f, indent=4)

        # actual_labels = [ann['category_id'] for ann in annotations]
        # actual_bboxes = [ann['bbox'] for ann in annotations]
        # for i, bbox in enumerate(actual_bboxes):
        #     x_min, y_min, w, h = bbox
        #     x_max = x_min + w
        #     y_max = y_min + h
        #     actual_bboxes[i] = [x_min, y_min, x_max, y_max]


        # batch = batch.transpose(1, 2, 0).copy()
        # batch = 255 * batch
        # batch = batch.astype(np.uint8)
        #
        # for bbox in predicted_bboxes:
        #     cv2.rectangle(batch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        #
        # for bbox in actual_bboxes:
        #     cv2.rectangle(batch, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

        #plt.imshow(batch)
        #plt.show()
        # from pycocotools.coco import COCO
        # from pycocotools.cocoeval import COCOeval
        #
        # coco = COCO("./coco/annotations/instances_val2017.json")
        #
        #
        # predicted_bboxes = np.array(predicted_bboxes)
        # predicted_bboxes[:, [0,1,2,3]] = predicted_bboxes[:, [0,2,1,3]]
        #
        # actual_bboxes = np.array(actual_bboxes)
        # actual_bboxes[:, [0, 1, 2, 3]] = actual_bboxes[:, [0, 2, 1, 3]]
        #
        # iou = complete_box_iou(torch.tensor(predicted_bboxes), torch.tensor(actual_bboxes))
        # found = (iou>0.2).nonzero(as_tuple=True)
        # pass
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

"""
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Load COCO annotations
coco = COCO("path/to/annotations.json")

# Load your predicted bounding boxes and class labels
pred_bboxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in zip(pred_x1, pred_y1, pred_x2, pred_y2)])
pred_labels = np.array(pred_labels)

# Run evaluation
coco_eval = COCOeval(coco, coco.loadRes(pred_bboxes, pred_labels), "bbox")
coco_eval.params.imgIds = image_ids # image_ids is a list of image ids to evaluate on
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# Extract mAP score
mAP = coco_eval.stats[0]
"""







if __name__ == "__main__":
    main()
