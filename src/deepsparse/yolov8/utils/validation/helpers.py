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
import argparse
import glob
import os
import warnings
from typing import List, Optional, Union

import yaml

import torch
from deepsparse.yolo import YOLOOutput as YOLODetOutput
from deepsparse.yolov8.schemas import YOLOSegOutput
from ultralytics.yolo.data.utils import ROOT


__all__ = ["data_from_dataset_path", "schema_to_tensor", "check_coco128_segmentation"]


def data_from_dataset_path(data: str, dataset_path: str) -> str:
    """
    Given a dataset name, fetch the yaml config for the dataset
    from the Ultralytics dataset repo, overwrite its 'path'
    attribute (dataset root dir) to point to the `dataset_path`
    and finally save it to the current working directory.
    This allows to create load data yaml config files that point
    to the arbitrary directories on the disk.

    :param data: name of the dataset (e.g. "coco.yaml")
    :param dataset_path: path to the dataset directory
    :return: a path to the new yaml config file
       (saved in the current working directory)
    """
    ultralytics_dataset_path = glob.glob(os.path.join(ROOT, "**", data), recursive=True)
    if len(ultralytics_dataset_path) != 1:
        raise ValueError(
            "Expected to find a single path to the "
            f"dataset yaml file: {data}, but found {ultralytics_dataset_path}"
        )
    ultralytics_dataset_path = ultralytics_dataset_path[0]
    with open(ultralytics_dataset_path, "r") as f:
        yaml_config = yaml.safe_load(f)
        yaml_config["path"] = dataset_path

        yaml_save_path = os.path.join(os.getcwd(), data)

        # save the new dataset yaml file
        with open(yaml_save_path, "w") as outfile:
            yaml.dump(yaml_config, outfile, default_flow_style=False)
        return yaml_save_path


def schema_to_tensor(
    pipeline_outputs: Union[YOLOSegOutput, YOLODetOutput], device: str
) -> List[Union[torch.Tensor, None, List[Optional[torch.Tensor]]]]:
    """
    Extract the intermediate outputs from the output schema.

    :param pipeline_outputs: YOLO output schema
    :param device: device to move the tensors to
    :return
        if segmentation,
            return [output, [None, None, mask_protos]]
        if detection,
            return [output, None]
        padding with Nones to match the input expected by the
        postprocess function in the validation script
    """

    preds = pipeline_outputs.intermediate_outputs
    if isinstance(preds, tuple):
        output, mask_protos = preds
        output = torch.from_numpy(output).to(device)
        mask_protos = torch.from_numpy(mask_protos).to(device)
        return [output, [None, None, mask_protos]]
    else:
        output = torch.from_numpy(preds).to(device)
        return [output, None]


def check_coco128_segmentation(args: argparse.Namespace):
    """
    Checks if the argument 'data' is coco128.yaml and if so,
    replaces it with coco128-seg.yaml.

    :param args: arguments to check
    :return: the updated arguments
    """
    if args.data == "coco128.yaml":
        dataset_name, dataset_extension = os.path.splitext(args.data)
        dataset_yaml = dataset_name + "-seg" + dataset_extension
        warnings.warn(
            f"Dataset yaml {dataset_yaml} is not supported for segmentation. "
            f"Attempting to use {dataset_yaml} instead."
        )
        args.data = dataset_yaml
    return args
