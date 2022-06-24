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
import numpy as np


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error


_all__ = ["detect", "decode", "postprocess", "sanitize_coordinates", "preprocess_array"]


def preprocess_array(
    image: numpy.ndarray, input_image_size: Tuple[int, int] = (550, 550)
) -> np.ndarray:
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
    # BGR -> RGB
    image = image[:, (2, 1, 0), :, :]
    image = numpy.ascontiguousarray(image)

    return image


def jaccard(
    box_a: numpy.ndarray, box_b: numpy.ndarray, iscrowd: bool = False
) -> numpy.ndarray:
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
    if box_a.ndim == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = numpy.expand_dims(
        (box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]), 2
    )  # [A,B]
    area_b = numpy.expand_dims(
        (box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]), 1
    )  # [A,B]
    area_a = numpy.broadcast_to(area_a, inter.shape)
    area_b = numpy.broadcast_to(area_b, inter.shape)
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else numpy.expand_dims(out, 0)


def sanitize_coordinates(
    _x1: numpy.ndarray, _x2: numpy.ndarray, img_size: int, padding: int = 0
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    Sanitizes the input coordinates so that
    x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    x1 = numpy.minimum(_x1, _x2)
    x2 = numpy.maximum(_x1, _x2)
    x1 = numpy.clip(x1 - padding, a_min=0, a_max=None)
    x2 = numpy.clip(x2 + padding, a_min=None, a_max=img_size)

    return x1, x2


def crop(masks: numpy.ndarray, boxes: numpy.ndarray, padding: int = 1) -> numpy.ndarray:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    "Crop" predicted masks by zeroing out everything not in the predicted bbox.

    Args:
        - masks should be a size [h, w, n] array of masks
        - boxes should be a size [n, 4] array of bbox coords in relative point form
    """
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = numpy.arange(w, dtype=x1.dtype).reshape(1, -1, 1)
    cols = numpy.arange(h, dtype=x1.dtype).reshape(-1, 1, 1)

    rows, cols = numpy.broadcast_to(rows, (h, w, n)), numpy.broadcast_to(
        cols, (h, w, n)
    )

    masks_left = rows >= x1.reshape(1, 1, -1)
    masks_right = rows < x2.reshape(1, 1, -1)
    masks_up = cols >= y1.reshape(1, 1, -1)
    masks_down = cols < y2.reshape(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask


def intersect(box_a: numpy.ndarray, box_b: numpy.ndarray) -> numpy.ndarray:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/box_utils.py

    We resize both arrays to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n, A, *_ = box_a.shape
    B = box_b.shape[1]
    max_xy = numpy.minimum(
        numpy.broadcast_to(numpy.expand_dims(box_a[:, :, 2:], 2), (n, A, B, 2)),
        numpy.broadcast_to(numpy.expand_dims(box_b[:, :, 2:], 1), (n, A, B, 2)),
    )
    min_xy = numpy.maximum(
        numpy.broadcast_to(numpy.expand_dims(box_a[:, :, :2], 2), (n, A, B, 2)),
        numpy.broadcast_to(numpy.expand_dims(box_b[:, :, :2], 1), (n, A, B, 2)),
    )
    return numpy.clip(max_xy - min_xy, a_min=0, a_max=None).prod(3)


def fast_nms(
    boxes: numpy.ndarray,
    masks: numpy.ndarray,
    scores: numpy.ndarray,
    confidence_threshold: float,
    max_num_detections: int,
    iou_threshold: float = 0.5,
    top_k: int = 200,
    second_threshold: bool = False,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/functions/detection.py
    """
    idx, scores = numpy.argsort(scores)[:, ::-1], numpy.sort(scores)[:, ::-1]

    num_classes, num_dets = idx.shape

    boxes = boxes[idx.flatten(), :].reshape(num_classes, num_dets, 4)
    masks = masks[idx.flatten(), :].reshape(num_classes, num_dets, -1)

    iou = jaccard(boxes, boxes)
    iou_max = numpy.amax(numpy.triu(iou, k=1), 1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_threshold

    scores = scores[:, :top_k] if top_k > keep.shape[1] else scores[:, : keep.shape[1]]
    # We should also only keep detections over the confidence threshold,
    # but at the cost of maxing out your detection count for every image,
    # you can just not do that. Because we have such a minimal amount of
    # computation per detection (matrix multiplication only), this increase
    # doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method,
    # you should do this second threshold.

    if second_threshold:
        keep *= scores > confidence_threshold

    # Assign each kept detection to its corresponding class
    classes = numpy.broadcast_to(numpy.arange(num_classes)[:, None], keep.shape)
    classes = classes[keep]

    boxes, masks, scores = boxes[keep], masks[keep], scores[keep]

    # Only keep the top cfg.max_num_detections the highest scores across all classes
    scores, idx = numpy.sort(scores)[::-1], numpy.argsort(scores)[::-1]
    idx = idx[:max_num_detections]
    scores = scores[:max_num_detections]

    classes = classes[idx]
    boxes = boxes[idx]
    masks = masks[idx]

    return boxes, masks, classes, scores


def decode(loc: numpy.ndarray, priors: numpy.ndarray) -> numpy.ndarray:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/functions/detection.py

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

    Returns: An array of decoded relative coordinates
        in point form with size [num_priors, 4]
    """

    variances = [0.1, 0.2]

    boxes = numpy.concatenate(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * numpy.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]

    return boxes


def detect(
    conf_preds: numpy.ndarray,
    decoded_boxes: numpy.ndarray,
    mask_data: numpy.ndarray,
    confidence_threshold: float,
    nms_threshold: float,
    max_num_detections,
    top_k: int,
) -> Dict[str, numpy.ndarray]:
    """
    Ported from
    https://github.com/neuralmagic/yolact/blob/master/layers/functions/detection.py

    Perform nms for only the max scoring class that isn't background (class 0)
    """
    conf_preds = conf_preds.transpose(1, 0)
    cur_scores = conf_preds[1:, :]
    conf_scores = numpy.max(cur_scores, axis=0)

    keep = conf_scores > confidence_threshold
    scores = cur_scores[:, keep]
    boxes = decoded_boxes[keep, :]
    masks = mask_data[keep, :]

    if scores.shape[0] == 0:
        return None

    boxes, masks, classes, scores = fast_nms(
        boxes,
        masks,
        scores,
        confidence_threshold,
        max_num_detections,
        nms_threshold,
        top_k,
    )

    return {"box": boxes, "mask": masks, "class": classes, "score": scores}


def postprocess(
    det_output: Dict[str, numpy.ndarray],
    crop_masks: bool = True,
    score_threshold: float = 0,
) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Ported from https://github.com/neuralmagic/yolact/blob/master/layers/output_utils.py

    Postprocess the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection
                                in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """

    detections = det_output

    if score_threshold > 0:
        keep = detections["score"] > score_threshold

        for k in detections:
            if k != "protos":
                detections[k] = detections[k][keep]

    classes, boxes, scores, masks, proto_data = (
        detections["class"],
        detections["box"],
        detections["score"],
        detections["mask"],
        detections["protos"],
    )

    masks = proto_data @ masks.T

    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))

    masks = sigmoid(masks)

    if crop_masks:
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.transpose(2, 0, 1)

    return classes, scores, boxes, masks


def _assert_channels_last(array: numpy.ndarray) -> np.ndarray:
    # make sure that the output is the array with dims
    # (B, H, W, C) or (H,W,C)
    if array.ndim == 4:
        if array.shape[1] < array.shape[2]:
            array = array.transpose(0, 2, 3, 1)
    else:
        if array.shape[0] < array.shape[1]:
            array = array.transpose(1, 2, 0)
    return array
