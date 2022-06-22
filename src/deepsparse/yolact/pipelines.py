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

from typing import Any, Dict, List, Tuple, Type, Union

import numpy
import torch
from pydantic import BaseModel

import cv2
from deepsparse import Pipeline
from deepsparse.yolact.schemas import YolactInputSchema, YolactOutputSchema
from deepsparse.utils import model_to_path

class Config:
    arbitrary_types_allowed = True

__all__ = ["YolactPipeline"]


@Pipeline.register(
    task="yolact",
    default_model_path=(
        None
    ),
)
class YolactPipeline(Pipeline):
    """
    An inference pipeline for YOLACT, encodes the preprocessing, inference and
    postprocessing for YOLACT models into a single callable object

    TODO: Fill Out Method Implementation
    """


    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.conf_thresh = 0.05
        self.nms_thresh = 0.5
        self.score_threshold = 0
        self.top_k = 200
        self.max_num_detections = 100

    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        return model_to_path(self.model_path)

    def process_inputs(
        self,
        inputs: BaseModel,
    ) -> List[numpy.ndarray]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_schema`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine. Can
            also include a tuple with engine inputs and special key word arguments
            to pass to process_engine_outputs to facilitate information from the raw
            inputs to postprocessing that may not be included in the engine inputs
        """
        images = inputs.images
        if not isinstance(images, list):
            images = [images]

        if isinstance(images[0], str):
            images = [cv2.imread(file_path) for file_path in images]

        return [self._process_numpy_array(array) for array in images]


    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        num_classes = 81
        h_image, w_image = (189, 266)
        import matplotlib.pyplot as plt
        boxes, confidence ,masks, priors, protos = engine_outputs
        batch_size, num_priors, _ = boxes.shape
        conf_preds = confidence.transpose(0,2,1)

        out = []

        for batch_idx in range(batch_size):
            decoded_boxes = self.decode(boxes[batch_idx], priors)
            result = self.detect(batch_idx, conf_preds, decoded_boxes, masks, None)

            if result is not None and protos is not None:
                result['proto'] = protos[batch_idx]

            out.append({'detection': result})

        t = self.postprocess(out, w=266, h=189, visualize_lincomb=False,
                        crop_masks=True,
                        score_threshold=0)
        top_k = 5
        class_color = False
        mask_alpha = 0.45
        undo_transform = False
        COLORS = ((244, 67, 54),
                  (233, 30, 99),
                  (156, 39, 176),
                  (103, 58, 183),
                  (63, 81, 181),
                  (33, 150, 243),
                  (3, 169, 244),
                  (0, 188, 212),
                  (0, 150, 136),
                  (76, 175, 80),
                  (139, 195, 74),
                  (205, 220, 57),
                  (255, 235, 59),
                  (255, 193, 7),
                  (255, 152, 0),
                  (255, 87, 34),
                  (121, 85, 72),
                  (158, 158, 158),
                  (96, 125, 139))

        idx = numpy.argsort(t[1])[::-1][:top_k]
        masks = t[3][idx]
        masks = masks.astype(numpy.float32)
        classes, scores, boxes = [x[idx] for x in t[:3]]


        ### this should be output I guess

        num_dets_to_consider = min(top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        # Quick and dirty lambda for selecting the color for a particular index
        # Also keeps track of a per-gpu color cache for maximum speed
        def get_color(j, on_gpu=None):
            global color_cache
            color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

            if on_gpu is not None and color_idx in color_cache[on_gpu]:
                return color_cache[on_gpu][color_idx]
            else:
                color = COLORS[color_idx]
                if not undo_transform:
                    # The image might come in as RGB or BRG, depending
                    color = torch.tensor((color[2], color[1], color[0]))
                if on_gpu is not None:
                    color = torch.Tensor(color).to(on_gpu).float() / 255.
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice

        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = numpy.concatenate([get_color(j, on_gpu=None).reshape(1, 1, 1, 3) for j in range(num_dets_to_consider)],             axis=0)
        masks_color = numpy.repeat(masks, 3, axis=3) * colors * mask_alpha


        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)


        img_numpy = cv2.imread('golfish.jpeg')
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
        img_numpy = img_numpy / 255


        img_numpy= img_numpy * inv_alph_masks.prod(axis=0) + masks_color_summand

        # Then draw the stuff that needs to be done on the cpu
        # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason

        display_bboxes = True
        display_text = True
        display_scores = True

        from deepsparse.yolo.utils.coco_classes import COCO_CLASSES


        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]
            color = [int(x) for x in color]
            if display_bboxes:
                img_numpy = cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if display_text:
                _class = COCO_CLASSES[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                img_numpy = cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                img_numpy = cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)

        return img_numpy









    @property
    def input_schema(self) -> Type[YolactInputSchema]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return YolactInputSchema

    @property
    def output_schema(self) -> Type[YolactOutputSchema]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return YolactOutputSchema

    def _process_numpy_array(self, image):
        import torch.nn.functional as F
        # We start with an input image (H, W, C)
        # and make sure it is type float
        image = image.astype(numpy.float32)
        image = cv2.resize(image, (550, 550))
        image = numpy.expand_dims(image, 0)
        # By default, when training we do not preserve
        # aspect ratio (simply explode the image size
        # to the default value)
        image = image.transpose(0, 3, 1, 2)
        image = numpy.ascontiguousarray(image)
        # Apply the default backbone transform
        image /= 255
        image = image[:, (2, 1, 0), :, :]

        return image

    def sanitize_coordinates(self, _x1, _x2, img_size: int, padding: int = 0, cast: bool = True):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.

        If cast is false, the result won't be cast to longs.
        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size
        x1 = numpy.minimum(_x1, _x2)
        x2 = numpy.maximum(_x1, _x2)
        x1 = numpy.clip(x1 - padding, a_max = x1 - padding, a_min=0)
        x2 = numpy.clip(x2 + padding, a_min = x2 + padding, a_max=img_size)

        return x1, x2


    def decode(self, loc, priors):
        """
        Decode predicted bbox coordinates using the same scheme
        employed by Yolov2: https://arxiv.org/pdf/1612.08242.pdf

            b_x = (sigmoid(pred_x) - .5) / conv_w + prior_x
            b_y = (sigmoid(pred_y) - .5) / conv_h + prior_y
            b_w = prior_w * exp(loc_w)
            b_h = prior_h * exp(loc_h)

        Note that loc is inputed as [(s(x)-.5)/conv_w, (s(y)-.5)/conv_h, w, h]
        while priors are inputed as [x, y, w, h] where each coordinate
        is relative to size of the image (even sigmoid(x)). We do this
        in the network by dividing by the 'cell size', which is just
        the size of the convouts.

        Also note that prior_x and prior_y are center coordinates which
        is why we have to subtract .5 from sigmoid(pred_x and pred_y).

        Args:
            - loc:    The predicted bounding boxes of size [num_priors, 4]
            - priors: The priorbox coords with size [num_priors, 4]

        Returns: A tensor of decoded relative coordinates in point form
                 form with size [num_priors, 4]
        """


        variances = [0.1, 0.2]

        boxes = numpy.concatenate((             priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],             priors[:, 2:] * numpy.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        return boxes

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores = numpy.max(cur_scores, axis=0)


        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]

        if scores.shape[0] == 0:
            return None

        boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
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

        inter = self.intersect(box_a, box_b)
        area_a = numpy.expand_dims((box_a[:, :, 2] - box_a[:, :, 0]) *
                  (box_a[:, :, 3] - box_a[:, :, 1]), 2)  # [A,B]
        area_b = numpy.expand_dims((box_b[:, :, 2] - box_b[:, :, 0]) *
                                   (box_b[:, :, 3] - box_b[:, :, 1]), 1)  # [A,B]
        area_a = numpy.broadcast_to(area_a, (inter.shape))
        area_b = numpy.broadcast_to(area_b, (inter.shape))
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else numpy.expand_dims(out, 0)

    def fast_nms(self, boxes, masks, scores, iou_threshold: float = 0.5, top_k: int = 200,
                 second_threshold: bool = False):
        idx = numpy.argsort(scores)[:,::-1]
        scores = numpy.sort(scores)[:,::-1] # descending False

        scores = scores[:, :top_k]

        num_classes, num_dets = idx.shape

        boxes = boxes[idx.flatten(), :].reshape(num_classes, num_dets, 4)
        masks = masks[idx.flatten(), :].reshape(num_classes, num_dets, -1)

        iou = self.jaccard(boxes, boxes)
        iou_max = numpy.amax(numpy.triu(iou,k=1),1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = numpy.broadcast_to(numpy.arange(num_classes)[:,None], keep.shape)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = numpy.sort(scores)[::-1], numpy.argsort(scores)[::-1]
        idx = idx[:self.max_num_detections]
        scores = scores[:self.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [n,A,4].
          box_b: (tensor) bounding boxes, Shape: [n,B,4].
        Return:
          (tensor) intersection area, Shape: [n,A,B].
        """
        n = box_a.shape[0]
        A = box_a.shape[1]
        B = box_b.shape[1]
        max_xy = numpy.minimum(
                           numpy.broadcast_to(numpy.expand_dims(box_a[:, :, 2:],2),(n, A, B, 2)),
                           numpy.broadcast_to(numpy.expand_dims(box_b[:, :, 2:], 1), (n, A, B, 2))
                           )
        min_xy = numpy.maximum(
                           numpy.broadcast_to(numpy.expand_dims(box_a[:, :, :2],2),(n, A, B, 2)),
                           numpy.broadcast_to(numpy.expand_dims(box_b[:, :, :2], 1), (n, A, B, 2))
        )
        return numpy.clip(max_xy - min_xy, a_min=0, a_max = None).prod(3)  # inter

    def postprocess(self, det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                    visualize_lincomb=False, crop_masks=True, score_threshold=0):
        """
        Postprocesses the output of Yolact on testing mode into a format that makes sense,
        accounting for all the possible configuration settings.

        Args:
            - det_output: The lost of dicts that Detect outputs.
            - w: The real with of the image.
            - h: The real height of the image.
            - batch_idx: If you have multiple images for this batch, the image's index in the batch.
            - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

        Returns 4 torch Tensors (in the following order):
            - classes [num_det]: The class idx for each detection.
            - scores  [num_det]: The confidence score for each detection.
            - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
            - masks   [num_det, h, w]: Full image masks for each detection.
        """

        dets = det_output[batch_idx]
        dets = dets['detection']

        if dets is None:
            return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

        if score_threshold > 0:
            keep = dets['score'] > score_threshold

            for k in dets:
                if k != 'proto':
                    dets[k] = dets[k][keep]

            if dets['score'].size(0) == 0:
                return [torch.Tensor()] * 4

        # Actually extract everything from dets now
        classes = dets['class']
        boxes = dets['box']
        scores = dets['score']
        masks = dets['mask']

        proto_data = dets['proto']

        masks = proto_data @ masks.T

        def sigmoid(x):
            return 1 / (1 + numpy.exp(-x))

        masks = sigmoid(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = self.crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.transpose(2, 0, 1)

        # Scale masks up to the full image
        # Damian -> not most efficient, to be revisited
        masks = numpy.stack([cv2.resize(mask,(w,h), interpolation=cv2.INTER_LINEAR) for mask in masks])

        # Binarize the masks
        masks = masks > 0.5

        boxes[:, 0], boxes[:, 2] = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
        boxes[:, 1], boxes[:, 3] = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
        boxes = boxes.astype(numpy.int64)

        return classes, scores, boxes, masks

    def crop(self, masks, boxes, padding: int = 1):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [h, w, n] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """
        h, w, n = masks.shape
        x1, x2 = self.sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
        y1, y2 = self.sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)

        rows = numpy.arange(w, dtype=x1.dtype).reshape(1, -1, 1)
        cols = numpy.arange(h, dtype=x1.dtype).reshape(-1, 1, 1)

        rows, cols = numpy.broadcast_to(rows, (h,w,n)), numpy.broadcast_to(cols, (h,w,n))

        masks_left = rows >= x1.reshape(1, 1, -1)
        masks_right = rows < x2.reshape(1, 1, -1)
        masks_up = cols >= y1.reshape(1, 1, -1)
        masks_down = cols < y2.reshape(1, 1, -1)

        crop_mask = masks_left * masks_right * masks_up * masks_down

        return masks * crop_mask

    def sanitize_coordinates(self, _x1, _x2, img_size: int, padding: int = 0, cast: bool = True):
        """
        Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
        Also converts from relative to absolute coordinates and casts the results to long tensors.

        If cast is false, the result won't be cast to longs.
        Warning: this does things in-place behind the scenes so copy if necessary.
        """
        _x1 = _x1 * img_size
        _x2 = _x2 * img_size
        if cast:
            _x1 = _x1.long()
            _x2 = _x2.long()
        x1 = numpy.minimum(_x1, _x2)
        x2 = numpy.maximum(_x1, _x2)
        x1 = numpy.clip(x1 - padding, a_min=0, a_max = None)
        x2 = numpy.clip(x2 + padding, a_min=None, a_max=img_size)

        return x1, x2