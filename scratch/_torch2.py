
import torch
import numpy
# from sparseml.pytorch.models.classification import resnet50
from torchvision.models import resnet50
from deepsparse.benchmark.torch_engine import TorchScriptEngine
from typing import List

print(0)
breakpoint()
model = resnet50(pretrained=True)
# engine = TorchEngine(model, device="cpu")
engine = TorchScriptEngine(model, device="cpu")
breakpoint()

inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
out = engine(inp)
assert isinstance(out, List) and all(isinstance(arr, numpy.ndarray) for arr in out)
print(1)


"""
Tests

* Torchengine with input args
* Torchengine call
* torch engine run




GPU
* allow the user to select the num of gpus?
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/torch_utils.py#L56



"""




