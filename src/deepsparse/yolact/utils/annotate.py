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
import numpy
import numpy as np

import cv2
import matplotlib.pyplot as plt
from deepsparse.yolact.schemas import YOLACTOutputSchema
from deepsparse.yolact.utils import sanitize_coordinates
from deepsparse.yolo.utils.coco_classes import COCO_CLASSES
from deepsparse.yolo.utils.utils import _get_color


def annotate_image(
    image: numpy.ndarray,
    prediction: YOLACTOutputSchema,
    score_threshold: float = 0.35,
    images_per_sec: float = 10,
):

    image_res = copy.copy(image)

    masks = prediction.masks[0]
    boxes = prediction.boxes[0]
    classes = prediction.classes[0]
    scores = prediction.scores[0]

    masks, boxes = _reshape_to_original(image, masks, boxes)
    for idx in range(len(boxes)):
        label = COCO_CLASSES[classes[idx]]
        if scores[idx] > score_threshold:
            colour = _get_color(classes[idx])
            left, top, _, _ = boxes[idx]
            image_res = _put_mask(image_res, mask=masks[idx], colour=colour)
            image_res = _put_bounding_box(
                image=image_res, box=boxes[idx], colour=colour
            )

            annotation_text = f"{label}: {scores[idx]:.0%}"
            text_width, text_height = _get_text_size(annotation_text)
            image_res = _put_annotation_text(
                image_res, annotation_text, left, top, colour, text_width, text_height
            )

    plt.imsave("wohoo.jpg", image_res)


import copy


def _put_mask(image, mask, colour):
    mask_coloured = np.where(mask[..., None], np.array(colour, dtype="uint8"), image)
    blended_image = cv2.addWeighted(image, 0.5, mask_coloured, 0.5, 0)
    return blended_image


def _put_annotation_text(
    image, annotation_text, left, top, colour, text_width, text_height
):
    image = cv2.rectangle(
        image,
        (int(left), int(top) - 33),
        (int(left) + text_width, int(top) - 28 + text_height),
        colour,
        thickness=-1,
    )

    image = cv2.putText(
        image,
        annotation_text,
        (int(left), int(top) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,  # font scale
        (255, 255, 255),  # white text
        2,  # thickness
        cv2.LINE_AA,
    )
    return image


def _put_bounding_box(image, box, colour):
    left, top, right, bottom = box
    image = cv2.rectangle(
        image,
        (int(left), int(top)),
        (int(right), int(bottom)),
        colour,
        thickness=1,
    )
    return image


def _get_text_size(annotation_text):
    (text_width, text_height), text_baseline = cv2.getTextSize(
        annotation_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,  # font scale
        2,  # thickness
    )
    text_height += text_baseline
    return text_width, text_height


def _reshape_to_original(original_image: numpy.ndarray, masks: numpy.ndarray, boxes):
    h, w, _ = original_image.shape

    # TODO: Is there a faster way of interpolating? scipy?
    # Resize the masks
    masks = numpy.stack(
        [cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) for mask in masks]
    )

    # Binarize the masks
    masks = (masks > 0.5).astype(numpy.int8)

    boxes = numpy.stack(boxes)
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
        boxes[:, 0],
        boxes[:, 2],
        w,
    )
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
    boxes = boxes.astype(numpy.int64)

    return masks, boxes


# class_color = False
#         mask_alpha = 0.45
#         undo_transform = False


#
#
# num_dets_to_consider = min(top_k, classes.shape[0])
#         for j in range(num_dets_to_consider):
#             if scores[j] < self.score_threshold:
#                 num_dets_to_consider = j
#                 break
#
#         # Quick and dirty lambda for selecting the color for a particular index
#         # Also keeps track of a per-gpu color cache for maximum speed
#         def get_color(j, on_gpu=None):
#             global color_cache
#             color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
#
#             if on_gpu is not None and color_idx in color_cache[on_gpu]:
#                 return color_cache[on_gpu][color_idx]
#             else:
#                 color = COLORS[color_idx]
#                 if not undo_transform:
#                     # The image might come in as RGB or BRG, depending
#                     color = torch.tensor((color[2], color[1], color[0]))
#                 if on_gpu is not None:
#                     color = torch.Tensor(color).to(on_gpu).float() / 255.0
#                     color_cache[on_gpu][color_idx] = color
#                 return color
#
#         # First, draw the masks on the GPU where we can do it really fast
#         # Beware: very fast but possibly unintelligible mask-drawing code ahead
#         # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
#
#         # After this, mask is of size [num_dets, h, w, 1]
#         masks = masks[:num_dets_to_consider, :, :, None]
#
#         # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
#         colors = numpy.concatenate(
#             [
#                 get_color(j, on_gpu=None).reshape(1, 1, 1, 3)
#                 for j in range(num_dets_to_consider)
#             ],
#             axis=0,
#         )
#         masks_color = numpy.repeat(masks, 3, axis=3) * colors * mask_alpha
#
#         # This is 1 everywhere except for 1-mask_alpha where the mask is
#         inv_alph_masks = masks * (-mask_alpha) + 1
#
#         # I did the math for this on pen and paper. This whole block should be equivalent to:
#         #    for j in range(num_dets_to_consider):
#         #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
#         masks_color_summand = masks_color[0]
#         if num_dets_to_consider > 1:
#             inv_alph_cumul = inv_alph_masks[: (num_dets_to_consider - 1)].cumprod(
#                 axis=0
#             )
#             masks_color_cumul = masks_color[1:] * inv_alph_cumul
#             masks_color_summand += masks_color_cumul.sum(axis=0)
#
#         img_numpy = cv2.imread("golfish.jpeg")
#         img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
#         img_numpy = img_numpy / 255
#
#         img_numpy = img_numpy * inv_alph_masks.prod(axis=0) + masks_color_summand
#
#         # Then draw the stuff that needs to be done on the cpu
#         # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
#
#         display_bboxes = True
#         display_text = True
#         display_scores = True
#
#         from deepsparse.yolo.utils.coco_classes import COCO_CLASSES
#
#         for j in reversed(range(num_dets_to_consider)):
#             x1, y1, x2, y2 = boxes[j, :]
#             color = get_color(j)
#             score = scores[j]
#             color = [int(x) for x in color]
#             if display_bboxes:
#                 img_numpy = cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)
#
#             if display_text:
#                 _class = COCO_CLASSES[classes[j]]
#                 text_str = "%s: %.2f" % (_class, score) if display_scores else _class
#
#                 font_face = cv2.FONT_HERSHEY_DUPLEX
#                 font_scale = 0.6
#                 font_thickness = 1
#
#                 text_w, text_h = cv2.getTextSize(
#                     text_str, font_face, font_scale, font_thickness
#                 )[0]
#
#                 text_pt = (x1, y1 - 3)
#                 text_color = [255, 255, 255]
#
#                 img_numpy = cv2.rectangle(
#                     img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1
#                 )
#                 img_numpy = cv2.putText(
#                     img_numpy,
#                     text_str,
#                     text_pt,
#                     font_face,
#                     font_scale,
#                     text_color,
#                     font_thickness,
#                     cv2.LINE_AA,
#                 )
#
#         return img_numpy
