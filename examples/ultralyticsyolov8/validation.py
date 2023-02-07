from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.cfg import get_cfg
from examples.ultralyticsyolov8.validation_utils.detection_validator import DetectionValidator
from deepsparse.yolo.utils import COCO_CLASSES
from deepsparse import Pipeline
from deepsparse.yolo import YOLOOutput, YOLOPipeline
from ultralytics.yolo.utils import ops
import torch

@Pipeline.register("yolov8")
class YOLOv8Pipeline(YOLOPipeline):
    def process_engine_outputs(
        self, engine_outputs, **kwargs
    ) -> YOLOOutput:
        # post-processing

        batch_output = engine_outputs[0]  # post-processed values stored in first output
        # NMS
        batch_output = ops.non_max_suppression(
            torch.from_numpy(batch_output),
            conf_thres=0.001,
            iou_thres=0.7,
            multi_label=True)

        batch_boxes, batch_scores, batch_labels = [], [], []

        for image_output in batch_output:
            batch_boxes.append(image_output[:, 0:4].tolist())
            batch_scores.append(image_output[:, 4].tolist())
            batch_labels.append(image_output[:, 5].tolist())
            if self.class_names is not None:
                batch_labels_as_strings = [
                    str(int(label)) for label in batch_labels[-1]
                ]
                batch_class_names = [
                    self.class_names[label_string]
                    for label_string in batch_labels_as_strings
                ]
                batch_labels[-1] = batch_class_names

        return YOLOOutput(
            boxes=batch_boxes,
            scores=batch_scores,
            labels=batch_labels,
        )

def val():
    pipeline = Pipeline.create("yolov8", model_path="yolov8n.onnx")

    # change data
    # instantiate pipeline
    args = get_cfg(DEFAULT_CFG)
    args.data = "coco128.yaml"

    validator = DetectionValidator(pipeline=pipeline, args=args)
    classes = {i: class_ for (i, class_) in enumerate(COCO_CLASSES)}
    validator(stride=32, model = "yolov8n.pt", classes = classes)



if __name__ == "__main__":
    val()