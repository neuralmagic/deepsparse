'''
Example created by Adriano A. Santos (https://github.com/adrianosantospb).
'''

# Imports
import models
from sparseml.pytorch.utils import ModuleExporter
import torch
from torch.utils.data import DataLoader

from models import *
from utils.utils import *

# Weights file
weights_file = "config/best.pt"
cfg_file = "config/yolov3-spp.cfg"
folder_to_save = "./models"
new_file_name = "yolov3-spp.onnx"

shape = (416,416)

# Create archived v3 model (random weights)
models.ONNX_EXPORT = True
model = models.Darknet(cfg_file, shape)

# Load weights
if weights_file.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights_file)['model'])
else:  # darknet format
    load_darknet_weights(model, weights_file)

# Export to ONNX with SparseML
exporter = ModuleExporter(model, folder_to_save)  # creates models directory to save to
exporter.export_onnx(torch.randn(1, 3, 416, 416), name=new_file_name)