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
Helpers and Utilities for YOLACT
"""
from typing import Dict, Tuple

import numpy

import torch


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error


_all__ = ["detect", "decode", "postprocess", "preprocess_array"]


def preprocess_array(
    image: numpy.ndarray, input_image_size: Tuple[int, int] = (550, 550)
) -> numpy.ndarray:
    """
    Preprocessing the input before feeding it into the YOLACT deepsparse pipeline

    :param image: numpy array representing input image(s). It can be batched (or not)
        and have an arbitrary dimensions order ((C,H,W) or (H,W,C)).
        It must have BGR channel order
    :param input_image_size: image size expected by the YOLACT network.
        Default is (550,550).
    :return: preprocessed numpy array (B, C, D, D); where (D,D) is image size expected
        by the network. It is a contiguous array with RGB channel order.
    """
    image = image.astype(numpy.float32)
    image = _assert_channels_last(image)
    if image.ndim == 4 and image.shape[:2] != input_image_size:
        image = numpy.stack([cv2.resize(img, input_image_size) for img in image])

    else:
        if image.shape[:2] != input_image_size:
            image = cv2.resize(image, input_image_size)
        image = numpy.expand_dims(image, 0)

    image = image.transpose(0, 3, 1, 2)
    image /= 255
    image = numpy.ascontiguousarray(image)

    return image


def jaccard(
    box_a: torch.Tensor, box_b: torch.Tensor, iscrowd: bool = False
) -> torch.Tensor:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    Compute the jaccard overlap of two sets of boxes. The jaccard overlap
    is simply the intersection over union of two boxes. Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(2)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


@torch.jit.script
def sanitize_coordinates(
    _x1: torch.Tensor,
    _x2: torch.Tensor,
    img_size: int,
    padding: int = 0,
    cast: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    Sanitizes the input coordinates so that
    x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates
    and casts the results to long tensors.
    If cast is false, the result won't be cast to longs.

    Warning:
    this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.long()
        _x2 = _x2.long()
    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2


@torch.jit.script
def crop(masks: torch.Tensor, boxes: torch.Tensor, padding: int = 1) -> torch.Tensor:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

    rows = (
        torch.arange(w, device=masks.device, dtype=x1.dtype)
        .view(1, -1, 1)
        .expand(h, w, n)
    )
    cols = (
        torch.arange(h, device=masks.device, dtype=x1.dtype)
        .view(-1, 1, 1)
        .expand(h, w, n)
    )

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask


@torch.jit.script
def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2),
    )
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter


def fast_nms(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    max_num_detections: int,
    nms_threshold: float = 0.5,
    top_k: int = 200,
    second_threshold: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/functions/detection.py
    """

    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones
    # higher than the threshold
    keep = iou_max <= nms_threshold

    # We should also only keep detections over
    # the confidence threshold, but at the cost of
    # maxing out your detection count for every image,
    # you can just not do that. Because we
    # have such a minimal amount of computation
    # per detection (matrix mulitplication only),
    # this increase doesn't affect us much
    # (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method,
    # you should do this second threshold.

    if second_threshold:
        keep *= scores > second_threshold

    # Assign each kept detection to its corresponding class
    classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    idx = idx[:max_num_detections]
    scores = scores[:max_num_detections]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores


@torch.jit.script
def decode(loc: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    Decode predicted bbox coordinates using the same scheme
    employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf
        b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
        b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
        b_w = prior_w * exp(loc_w)
        b_h = prior_h * exp(loc_h)

    Note that loc is input as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
    while priors are input as [x, y, w, h] where each coordinate
    is relative to size of the image (even sigmoid(x)). We do this
    in the network by dividing by the 'cell size', which is just
    the size of the conv-outs.

    Also note that prior_x and prior_y are center coordinates which
    is why we have to subtract .5 from sigmoid(pred_x and pred_y).

    Args:
        - loc:    The predicted bounding boxes of size [num_priors, 4]
        - priors: The prior box coords with size [num_priors, 4]

    Returns: A tensor of decoded relative coordinates in point
             form with size [num_priors, 4]
    """

    variances = [0.1, 0.2]

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def detect(
    confidence_single_image: torch.Tensor,
    decoded_boxes: torch.Tensor,
    masks_single_image: torch.Tensor,
    confidence_threshold: float,
    nms_threshold: float,
    max_num_detections: int,
    top_k: int,
) -> Dict[str, torch.Tensor]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/functions/detection.py

    Perform nms for only the max scoring class that isn't background (class 0)
    """

    conf_preds = confidence_single_image.T
    cur_scores = conf_preds[1:, :]
    conf_scores, _ = torch.max(cur_scores, dim=0)

    keep = conf_scores > confidence_threshold
    scores = cur_scores[:, keep]
    boxes = decoded_boxes[keep, :]
    masks = masks_single_image[keep, :]

    if scores.size(1) == 0:
        return None

    boxes, masks, classes, scores = fast_nms(
        boxes,
        masks,
        scores,
        max_num_detections,
        nms_threshold,
        top_k,
    )

    return {"box": boxes, "mask": masks, "class": classes, "score": scores}


def postprocess(
    dets: Dict[str, torch.Tensor], crop_masks: bool = True, score_threshold: float = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/output_utils.py

    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.
    Args:
        - dets: The lost of dicts that Detect outputs.

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection
            in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    if score_threshold > 0:
        keep = dets["score"] > score_threshold

        for k in dets:
            if k != "protos":
                dets[k] = dets[k][keep]

    classes = dets["class"]
    boxes = dets["box"]
    scores = dets["score"]
    masks = dets["mask"]
    proto_data = dets["protos"]

    masks = proto_data @ masks.t()
    masks = torch.sigmoid(masks)

    if crop_masks:
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()

    return classes, scores, boxes, masks


def _assert_channels_last(array: numpy.ndarray) -> numpy.ndarray:
    # make sure that the output is an array with dims
    # (B, H, W, C) or (H,W,C)
    if array.ndim == 4:
        if (array.shape[1] < array.shape[2]) and (array.shape[1] < array.shape[3]):
            # if (B, C, W, H) then swap channels if
            # C < W and C < W
            array = array.transpose(0, 2, 3, 1)
    else:
        if (array.shape[0] < array.shape[1]) and (array.shape[0] < array.shape[2]):
            # if (C, W, H) then swap channels if
            # C < W and C < H
            array = array.transpose(1, 2, 0)
    return array
