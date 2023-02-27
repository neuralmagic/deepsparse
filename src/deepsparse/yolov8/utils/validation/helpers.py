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

    preds = pipeline_outputs.intermediate_outputs

    output, mask_protos = preds
    output = torch.from_numpy(output).to(device)
    mask_protos = torch.from_numpy(mask_protos).to(device)
    return [output, [None, None, mask_protos]]
