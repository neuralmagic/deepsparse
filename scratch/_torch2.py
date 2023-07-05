
import torch
import numpy
# from sparseml.pytorch.models.classification import resnet50
from torchvision.models import resnet50
from deepsparse.benchmark.torch_engine import TorchScriptEngine
from typing import List

model = resnet50(pretrained=True)
engine = TorchScriptEngine(model, device="cpu")

inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
out = engine(inp)
assert isinstance(out, List) and all(isinstance(arr, numpy.ndarray) for arr in out)
print(1)



