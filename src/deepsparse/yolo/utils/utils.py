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
Helpers and Utilities for YOLO
"""
import functools
import itertools
import logging
import random
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional, Tuple, Union

import numpy
import onnx
import torchvision
import yaml

import torch
from deepsparse.yolo.schemas import YOLOOutput


try:
    import cv2

    cv2_error = None
except ModuleNotFoundError as cv2_import_error:
    cv2 = None
    cv2_error = cv2_import_error

_YOLO_CLASS_COLORS = list(itertools.product([0, 255, 128, 64, 192], repeat=3))
_YOLO_CLASS_COLORS.remove((255, 255, 255))  # remove white from possible colors
_LOGGER = logging.getLogger(__name__)

# Default YOLO anchor grids
_YOLO_DEFAULT_ANCHORS = [
    torch.Tensor([[10, 13], [16, 30], [33, 23]]),
    torch.Tensor([[30, 61], [62, 45], [59, 119]]),
    torch.Tensor([[116, 90], [156, 198], [373, 326]]),
]
_YOLO_DEFAULT_ANCHOR_GRIDS = [
    t.clone().view(1, -1, 1, 1, 2) for t in _YOLO_DEFAULT_ANCHORS
]


@functools.lru_cache(maxsize=None)
def _get_color(label):
    # cache color lookups
    return random.choice(_YOLO_CLASS_COLORS)


class YoloPostprocessor:
    """
    Class for performing post-processing of YOLO model predictions

    :param image_size: size of input image to model. used to calculate stride based on
        output shapes
    """

    def __init__(
        self, image_size: Tuple[int, int] = (640, 640), cfg: Optional[str] = None
    ):
        self._image_size = image_size
        self._anchor_grids = (
            self._load_cfg_anchor_grid(cfg) if cfg else _YOLO_DEFAULT_ANCHOR_GRIDS
        )
        self._grids = {}  # Dict[Tuple[int], torch.Tensor]

    def pre_nms_postprocess(self, outputs: List[numpy.ndarray]) -> torch.Tensor:
        """
        :param outputs: raw outputs of a YOLO model before anchor grid processing
        :return: post-processed model outputs without NMS.
        """
        # postprocess and transform raw outputs into single torch tensor
        processed_outputs = []
        for idx, pred in enumerate(outputs):
            pred = torch.from_numpy(pred)
            pred = pred.sigmoid()

            # get grid and stride
            grid_shape = pred.shape[2:4]
            grid = self._get_grid(grid_shape)
            stride = self._image_size[0] / grid_shape[0]

            # decode xywh box values
            pred[..., 0:2] = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
            pred[..., 2:4] = (pred[..., 2:4] * 2) ** 2 * self._anchor_grids[idx]
            # flatten anchor and grid dimensions ->
            #       (bs, num_predictions, num_classes + 5)
            processed_outputs.append(pred.view(pred.size(0), -1, pred.size(-1)))
        return torch.cat(processed_outputs, 1)

    def _get_grid(self, grid_shape: Tuple[int, int]) -> torch.Tensor:
        if grid_shape not in self._grids:
            # adapted from yolov5.yolo.Detect._make_grid
            coords_y, coords_x = torch.meshgrid(
                [torch.arange(grid_shape[0]), torch.arange(grid_shape[1])]
            )
            grid = torch.stack((coords_x, coords_y), 2)
            self._grids[grid_shape] = grid.view(
                1, 1, grid_shape[0], grid_shape[1], 2
            ).float()
        return self._grids[grid_shape]

    @staticmethod
    def _load_cfg_anchor_grid(cfg: str) -> List[torch.Tensor]:
        with open(cfg) as f:
            anchors = yaml.safe_load(f)["anchors"]

        def _split_to_coords(coords_list):
            return [
                [coords_list[idx], coords_list[idx + 1]]
                for idx in range(0, len(coords_list), 2)
            ]

        anchors = [torch.Tensor(_split_to_coords(coords)) for coords in anchors]
        return [t.clone().view(1, -1, 1, 1, 2) for t in anchors]


def postprocess_nms(
    outputs: Union[torch.Tensor, numpy.ndarray],
    iou_thres: float = 0.25,
    conf_thres: float = 0.45,
    multi_label: bool = False,
) -> List[numpy.ndarray]:
    """
    :param outputs: Tensor of post-processed model outputs
    :param iou_thres: minimum IoU for a detection to be valid
    :param conf_thres: minimum confidence score for a detection to be valid
    :return: List of numpy arrays of NMS predictions for each image in the batch
    """
    # run nms in PyTorch, only post-process first output
    if isinstance(outputs, numpy.ndarray):
        outputs = torch.from_numpy(outputs)
    nms_outputs = _non_max_suppression(
        outputs, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label
    )
    return [output.cpu().numpy() for output in nms_outputs]


def _non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
):
    # Ported from ultralytics/yolov5

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, (
        f"Invalid Confidence threshold {conf_thres}, "
        "valid values are between 0.0 and 1.0"
    )
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    _, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.perf_counter()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            label_ = labels[xi]
            v = torch.zeros((len(label_), nc + 5), device=x.device)
            v[:, :4] = label_[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(label_)), label_[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = _xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = _box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.perf_counter() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def _xywh2xyxy(
    x: Union[torch.Tensor, numpy.ndarray]
) -> Union[torch.Tensor, numpy.ndarray]:
    # ported from ultralytics/yolov5
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    # where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (
            torch.min(box1[:, None, 2:], box2[:, 2:])
            - torch.max(box1[:, None, :2], box2[:, :2])
        )
        .clamp(0)
        .prod(2)
    )
    return inter / (
        area1[:, None] + area2 - inter
    )  # iou = inter / (area1 + area2 - inter)


def yolo_onnx_has_postprocessing(model_path: Union[str, onnx.ModelProto]) -> bool:
    """
    :param model_path: file path to YOLO ONNX model or loaded model
    :return: True if YOLO postprocessing (pre-nms) is included in the ONNX graph,
        this is assumed to be when the first output of the model has fewer dimensions
        than the other outputs as the grid dimensions have been flattened
    """
    if isinstance(model_path, str):
        model = onnx.load(model_path)
    else:
        model = model_path

    # get number of dimensions in each output
    outputs_num_dims = [
        len(output.type.tensor_type.shape.dim) for output in model.graph.output
    ]

    # assume if only one output, then it is post-processed
    if len(outputs_num_dims) == 1:
        return True

    return all(num_dims > outputs_num_dims[0] for num_dims in outputs_num_dims[1:])


def get_onnx_expected_image_shape(onnx_model: onnx.ModelProto) -> Tuple[int, ...]:
    """
    :param onnx_model: onnx model to get expected image shape of
    :return: expected shape of the input tensor from onnx graph as a 2-tuple
    """
    input_tensor = onnx_model.graph.input[0]
    return (
        input_tensor.type.tensor_type.shape.dim[2].dim_value,
        input_tensor.type.tensor_type.shape.dim[3].dim_value,
    )


def modify_yolo_onnx_input_shape(
    model_path: str, image_shape: Tuple[int, int]
) -> Tuple[str, Optional[NamedTemporaryFile]]:
    """
    Creates a new YOLO ONNX model from the given path that accepts the given input
    shape. If the given model already has the given input shape no modifications are
    made. Uses a tempfile to store the modified model file.

    :param model_path: file path to YOLO ONNX model
    :param image_shape: 2-tuple of the image shape to resize this yolo model to
    :return: filepath to an onnx model reshaped to the given input shape will be the
        original path if the shape is the same.  Additionally returns the
        NamedTemporaryFile for managing the scope of the object for file deletion
    """
    has_postprocessing = yolo_onnx_has_postprocessing(model_path)

    model = onnx.load(model_path)
    model_input = model.graph.input[0]

    initial_x, initial_y = get_onnx_expected_image_shape(model)

    if not (isinstance(initial_x, int) and isinstance(initial_y, int)):
        return model_path, None  # model graph does not have static integer input shape

    if (initial_x, initial_y) == tuple(image_shape):
        return model_path, None  # no shape modification needed

    # override input shape
    model_input.type.tensor_type.shape.dim[2].dim_value = image_shape[0]
    model_input.type.tensor_type.shape.dim[3].dim_value = image_shape[1]

    # override output shape to account for stride
    scale_x = initial_x / image_shape[0]
    scale_y = initial_y / image_shape[1]

    for idx, model_output in enumerate(model.graph.output):
        if idx == 0 and has_postprocessing:
            continue
        output_x = get_tensor_dim_shape(model_output, 2)
        output_y = get_tensor_dim_shape(model_output, 3)
        set_tensor_dim_shape(model_output, 2, int(output_x / scale_x))
        set_tensor_dim_shape(model_output, 3, int(output_y / scale_y))

    # fix number of predictions in post-processed output for new strides
    if has_postprocessing:
        # sum number of predictions across the other outputs
        num_predictions = sum(
            numpy.prod(
                [
                    get_tensor_dim_shape(output_tensor, dim_idx)
                    for dim_idx in range(1, 4)
                ]
            )
            for output_tensor in model.graph.output[1:]
        )
        set_tensor_dim_shape(model.graph.output[0], 1, num_predictions)

    tmp_file = NamedTemporaryFile()  # file will be deleted after program exit
    onnx.save(model, tmp_file.name)

    return tmp_file.name, tmp_file


def get_tensor_dim_shape(tensor: onnx.TensorProto, dim: int) -> int:
    """
    :param tensor: ONNX tensor to get the shape of a dimension of
    :param dim: dimension index of the tensor to get the shape of
    :return: shape of the tensor at the given dimension
    """
    return tensor.type.tensor_type.shape.dim[dim].dim_value


def set_tensor_dim_shape(tensor: onnx.TensorProto, dim: int, value: int):
    """
    Sets the shape of the tensor at the given dimension to the given value

    :param tensor: ONNX tensor to modify the shape of
    :param dim: dimension index of the tensor to modify the shape of
    :param value: new shape for the given dimension
    """
    tensor.type.tensor_type.shape.dim[dim].dim_value = value


def annotate_image(
    image: numpy.ndarray,
    prediction: YOLOOutput,
    images_per_sec: Optional[float] = None,
    score_threshold: float = 0.35,
    model_input_size: Tuple[int, int] = None,
) -> numpy.ndarray:
    """
    Draws bounding boxes on predictions of a detection model

    :param image: original image to annotate (no pre-processing needed)
    :param prediction: predictions returned by the inference pipeline
    :param images_per_sec: optional fps value to annotate the left corner
        of the image (video) with
    :param score_threshold: minimum score a detection should have to be annotated
        on the image. Default is 0.35
    :param model_input_size: 2-tuple of expected input size for the given model to
        be used for bounding box scaling with original image. Scaling will not
        be applied if model_input_size is None. Default is None
    :return: the original image annotated with the given bounding boxes
    """
    boxes = prediction[0].boxes
    scores = prediction[0].scores
    labels = prediction[0].labels

    img_res = numpy.copy(image)

    scale_y = image.shape[0] / (1.0 * model_input_size[0]) if model_input_size else 1.0
    scale_x = image.shape[1] / (1.0 * model_input_size[1]) if model_input_size else 1.0

    for idx in range(len(boxes)):
        label = labels[idx]
        if scores[idx] > score_threshold:
            annotation_text = f"{label}: {scores[idx]:.0%}"

            # bounding box points
            left = boxes[idx][0] * scale_x
            top = boxes[idx][1] * scale_y
            right = boxes[idx][2] * scale_x
            bottom = boxes[idx][3] * scale_y

            # calculate text size
            (text_width, text_height), text_baseline = cv2.getTextSize(
                annotation_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                2,  # thickness
            )
            text_height += text_baseline

            # make solid background for annotation text
            cv2.rectangle(
                img_res,
                (int(left), int(top) - 33),
                (int(left) + text_width, int(top) - 28 + text_height),
                _get_color(label),
                thickness=-1,  # filled solid
            )

            # add white annotation text
            cv2.putText(
                img_res,
                annotation_text,
                (int(left), int(top) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                (255, 255, 255),  # white text
                2,  # thickness
                cv2.LINE_AA,
            )

            # draw bounding box
            cv2.rectangle(
                img_res,
                (int(left), int(top)),
                (int(right), int(bottom)),
                _get_color(label),
                thickness=2,
            )

    if images_per_sec is not None:
        img_res = _plot_fps(
            img_res=img_res,
            images_per_sec=images_per_sec,
            x=20,
            y=30,
            font_scale=0.9,
            thickness=2,
        )
    return img_res


def _plot_fps(
    img_res: numpy.ndarray,
    images_per_sec: float,
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> numpy.ndarray:

    annotation_text = f"FPS: {int(images_per_sec)}"
    # calculate text size
    (text_width, text_height), text_baseline = cv2.getTextSize(
        annotation_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,  # font scale
        thickness,  # thickness
    )
    # make solid background for annotation text
    cv2.rectangle(
        img_res,
        (x, y - 3 * text_baseline),
        (x + text_width, y + text_height - text_baseline),
        (255, 255, 255),
        thickness=-1,  # filled solid
    )

    cv2.putText(
        img_res,
        annotation_text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 46, 6),  # color
        thickness,
        cv2.LINE_AA,
    )
    return img_res
