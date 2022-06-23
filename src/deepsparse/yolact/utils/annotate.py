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

class_color = False
        mask_alpha = 0.45
        undo_transform = False
        COLORS = (
            (244, 67, 54),
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
            (96, 125, 139),
        )


masks = numpy.stack(
            [cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) for mask in masks]
        )

        # Binarize the masks
        masks = masks > 0.5

        boxes[:, 0], boxes[:, 2] = sanitize_coordinates(
            boxes[:, 0],
            boxes[:, 2],
            w,
        )
        boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h)
        boxes = boxes.astype(numpy.int64)

        return classes, scores, boxes, masks


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
                    color = torch.Tensor(color).to(on_gpu).float() / 255.0
                    color_cache[on_gpu][color_idx] = color
                return color

        # First, draw the masks on the GPU where we can do it really fast
        # Beware: very fast but possibly unintelligible mask-drawing code ahead
        # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice

        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = numpy.concatenate(
            [
                get_color(j, on_gpu=None).reshape(1, 1, 1, 3)
                for j in range(num_dets_to_consider)
            ],
            axis=0,
        )
        masks_color = numpy.repeat(masks, 3, axis=3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[: (num_dets_to_consider - 1)].cumprod(
                axis=0
            )
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        img_numpy = cv2.imread("golfish.jpeg")
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
        img_numpy = img_numpy / 255

        img_numpy = img_numpy * inv_alph_masks.prod(axis=0) + masks_color_summand

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
                text_str = "%s: %.2f" % (_class, score) if display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(
                    text_str, font_face, font_scale, font_thickness
                )[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                img_numpy = cv2.rectangle(
                    img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1
                )
                img_numpy = cv2.putText(
                    img_numpy,
                    text_str,
                    text_pt,
                    font_face,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        return img_numpy