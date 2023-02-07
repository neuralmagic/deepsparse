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

import click

from deepsparse import Pipeline
from deepsparse.yolo.utils import COCO_CLASSES
from deepsparse.yolov8.utils import DetectionValidator
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
SUPPORTED_DATASET_CONFIGS = ["coco128.yaml", "coco.yaml"]


@click.command()
@click.option(
    "--model-path",
    "--model_path",
    required=True,
    help="Path to SparseZoo/onnx model or SparseZoo stub f"
    "or the YOLOv8 model to be evaluated.",
)
@click.option(
    "--dataset-yaml",
    "--dataset_yaml",
    type=str,
    default="coco128.yaml",
    show_default=True,
    help="Dataset yaml supported by the ultralytics framework. "
    "Could be e.g. `coco128.yaml` or `coco.yaml`. "
    "Defaults to `coco128.yaml",
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
    "--batch-size",
    "--batch_size",
    type=int,
    default=16,
    show_default=True,
    help="Validation batch size",
)
@click.option(
    "--stride",
    type=int,
    default=32,
    show_default=True,
    help="YOLOv8 can handle arbitrary sized images as long as "
    "both sides are a multiple of 32. This is because the "
    "maximum stride of the backbone is 32 and it is a fully "
    "convolutional network.",
)
@click.option(
    "--engine_type",
    default=DEEPSPARSE_ENGINE,
    type=click.Choice([DEEPSPARSE_ENGINE, ORT_ENGINE]),
    show_default=True,
    help="engine type to use, valid choices: ['deepsparse', 'onnxruntime']",
)
@click.option(
    "--device",
    default="0",
    type=str,
    show_default=True,
    help="Use 'device=cpu' or pass valid CUDA device(s) if available, "
    "i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU",
)
def main(
    dataset_yaml: str,
    model_path: str,
    batch_size: int,
    num_cores: int,
    engine_type: str,
    stride: int,
    device: str,
):

    pipeline = Pipeline.create(
        task="yolov8",
        model_path=model_path,
        num_cores=num_cores,
        engine_type=engine_type,
    )

    args = get_cfg(DEFAULT_CFG)
    args.data = dataset_yaml
    args.batch_size = batch_size
    print(args.device)
    args.device = device
    print(args.device)

    if dataset_yaml not in SUPPORTED_DATASET_CONFIGS:
        raise ValueError(
            f"Dataset yaml {dataset_yaml} is not supported. "
            f"Supported dataset configs are {SUPPORTED_DATASET_CONFIGS})"
        )

    classes = {label: class_ for (label, class_) in enumerate(COCO_CLASSES)}

    validator = DetectionValidator(pipeline=pipeline, args=args)
    validator(stride=stride, classes=classes)


if __name__ == "__main__":
    main()
