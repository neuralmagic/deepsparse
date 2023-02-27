from typing import Union, List
import torch
from deepsparse.yolo import YOLOOutput as YOLODetOutput
from deepsparse.yolov8.schemas import YOLOSegOutput

__all__ = ["schema_to_tensor"]

def schema_to_tensor(pipeline_outputs: Union[YOLOSegOutput, YOLODetOutput], device: str) -> List[torch.Tensor]:
    """
    Transform the YOLOOutput to the format expected by the validation code.

    :param pipeline_outputs: YOLOOutput from the pipeline
    :param device: device to move the tensors to
    :return list of tensor with the format [x1, y1, x2, y2, confidence, class]
    """

    preds = []

    for boxes, labels, confidence in zip(
        pipeline_outputs.boxes, pipeline_outputs.labels, pipeline_outputs.scores
    ):

        boxes = torch.tensor(boxes)

        # map labels to integers and reshape for concatenation
        labels = list(map(int, list(map(float, labels))))
        labels = torch.tensor(labels).view(-1, 1)

        # reshape for concatenation
        scores = torch.tensor(confidence).view(-1, 1)
        # concatenate and append to preds
        preds.append(torch.cat([boxes, scores, labels], axis=1).to(device))
    return preds