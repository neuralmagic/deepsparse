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

import os
import re
from typing import Optional

import torchvision.models as models


try:
    import torch

    torch_import_error = None
except Exception as torch_import_err:
    torch_import_error = torch_import_err
    torch = None


def find_file_with_pattern(folder_path: str, pattern: str) -> Optional[str]:
    folder_path = os.path.expanduser(folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if re.match(pattern, file):
                return os.path.join(root, file)


def save_pth_to_pt(model_path_pth: str, model_name: str = "resnet50") -> None:
    """
    Given .pth model path, load and save model as .pt
    """
    model_func = getattr(models, model_name)
    model = model_func()
    model.load_state_dict(
        torch.load(model_path_pth)
    )  # Load the saved weights into the model
    scripted_model = torch.jit.script(model)

    model_name = model_path_pth.split(".pth")
    model_path_pt = f"{model_name[0]}.pt"
    torch.jit.save(scripted_model, model_path_pt)
    scripted_model.save(model_path_pt)
    return model_path_pt
