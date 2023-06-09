import torch
import numpy

from deepsparse.benchmark.torch_engine import TorchEngine


path = "/home/george/model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path, map_location=device)


print(type(model))
print(model)

# path = "zoo:cv/classification/mobilxenet_v1-1.0/pytorch/sparseml/imagenet/pruned_quant-moderate"

t = TorchEngine(model=model)
print(t)

arr = [numpy.zeros((3, 28))]

t(arr)



"""

TODO:

self._model_path = model_to_path(model)  # None if pass an actual Module
    needs to return .pt model, not onnx


Ben Notes:

# from torchvision.models import resnet50
# import torch

# model = resnet50()
# torch.jit.save(model, "resnet50.pt")
# import torch
# model = torch.load("resnet50.pt")


from sparseml.pytorch.models import resnet50
import torch

model = resnet50(pretrained=True)
torch.jit.save(model, "resnet50.pt")
import torch
model = torch.load("resnet50.pt")


---
check version of torch for cuda compatibility



"""
