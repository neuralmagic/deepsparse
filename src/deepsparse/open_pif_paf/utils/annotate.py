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

try:
    import openpifpaf

    openpifpaf_error = None
except ModuleNotFoundError as openpifpaf_import_error:
    openpifpaf = None
    openpifpaf_error = openpifpaf_import_error

import logging
from typing import List, Optional, Tuple

import numpy

import cv2
import torch
from deepsparse.open_pif_paf.schemas import OpenPifPafOutput
from deepsparse.yolact.utils.annotate import (
    _get_text_size,
    _plot_fps,
    _put_annotation_text,
)
from openpifpaf import decoder, network


_LOGGER = logging.getLogger(__name__)

# necessary openpifpaf dependencies for now
model_cpu, _ = network.Factory().factory(head_metas=None)
processor = decoder.factory(model_cpu.head_metas)


def _get_color():
    return tuple(numpy.random.randint(0, 256, 3).tolist())


# assign a color to a group of body parts
BODY_PART_COLOR = {
    "ankle_knee": _get_color(),
    "knee_hip": _get_color(),
    "shoulder_elbow": _get_color(),
    "elbow_wrist": _get_color(),
    "nose_eye:": _get_color(),
    "eye_ear": _get_color(),
    "ear_shoulder": _get_color(),
    "other": _get_color(),
}

CONFIDENCE_BOX_COLOR = _get_color()


def annotate_image(
    image: numpy.ndarray,
    prediction: OpenPifPafOutput,
    images_per_sec: Optional[float] = None,
    **kwargs,
) -> numpy.ndarray:

    cafs, cifs = prediction.caf, prediction.cif
    # we expect to have received batch = 1 from the pipeline
    if not (cafs.shape[0] == 1 and cifs.shape[0] == 1):
        raise ValueError("Expected batch size of 1, got {}".format(cafs.shape[0]))
    annotations = processor._mappable_annotations(
        [torch.tensor(cifs[0]), torch.tensor(cafs[0])], None, None
    )

    input_image_resolution = image.shape[:2]
    scale = numpy.flipud(
        numpy.divide(input_image_resolution, kwargs["model_resolution"])
    )

    for annotation in annotations:
        data = numpy.multiply(annotation.data[:, :2], scale)
        keypoints = annotation.keypoints
        image = _draw_skelethon(image, annotation.skeleton, data, keypoints)
        image = _draw_joints(image, data)
        image = _draw_confidence(image, data[0], annotation.score)
        if images_per_sec:
            image = _plot_fps(
                img_res=image,
                images_per_sec=images_per_sec,
                x=20,
                y=30,
                font_scale=0.9,
                thickness=2,
            )
    return image


def _draw_confidence(
    image: numpy.ndarray, coords: numpy.ndarray, score: float
) -> numpy.ndarray:
    """
    Draw the confidence score of the pose estimation
    """
    annotation_text = f"{score:.0%}"
    text_width, text_height = _get_text_size(annotation_text)
    image = _put_annotation_text(
        image=image,
        annotation_text=annotation_text,
        left=coords[0],
        top=coords[1],
        color=CONFIDENCE_BOX_COLOR,
        text_width=text_width,
        text_height=text_height,
    )
    return image


def _draw_joints(
    image: numpy.ndarray,
    data: numpy.ndarray,
    joint_thickness: int = 8,
    trace_thickness: int = 2,
) -> numpy.ndarray:
    """
    Draw the joints of the pose estimation
    """
    for joint in data:
        x, y = joint
        image = cv2.circle(
            image, (int(x), int(y)), joint_thickness + trace_thickness, (0, 0, 0), -1
        )
        image = cv2.circle(
            image, (int(x), int(y)), joint_thickness, (255, 255, 255), -1
        )
    return image


def _draw_skelethon(
    image: numpy.ndarray,
    skelethon: List[Tuple[int, int]],
    data: numpy.ndarray,
    keypoints: List[str],
    connection_thickness: int = 8,
    trace_thickness: int = 2,
) -> numpy.ndarray:
    """
    Draw the skeleton of the pose estimation
    """
    for connection in skelethon:
        joint_1 = connection[0] - 1
        joint_2 = connection[1] - 1
        keypoint_1, keypoint_2 = data[joint_1], data[joint_2]
        if all(keypoint_1) is False or all(keypoint_2) is False:
            continue
        body_part_1_name, body_part_2_name = keypoints[joint_1], keypoints[joint_2]
        color = _get_connection_color(body_part_1_name, body_part_2_name)
        x1, y1 = keypoint_1[0], keypoint_1[1]
        x2, y2 = keypoint_2[0], keypoint_2[1]
        image = cv2.line(
            image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 0, 0),
            connection_thickness + trace_thickness,
        )
        image = cv2.line(
            image, (int(x1), int(y1)), (int(x2), int(y2)), color, connection_thickness
        )
    return image


def _get_connection_color(
    body_part_name1: str, body_part_name2: str, body_part_color_dict=BODY_PART_COLOR
):
    """
    Get the label color of the connection between two body parts
    """
    if "_" in body_part_name1:
        body_part_name1 = body_part_name1.split("_")[-1]
    if "_" in body_part_name2:
        body_part_name2 = body_part_name2.split("_")[-1]

    color = body_part_color_dict.get(body_part_name1 + "_" + body_part_name2)
    if color is None:
        color = body_part_color_dict.get(
            body_part_name2 + "_" + body_part_name1, body_part_color_dict["other"]
        )
    return color
