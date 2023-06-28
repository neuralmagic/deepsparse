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

from typing import List
from unittest.mock import Mock

import numpy
import os
import pytest
from deepsparse.benchmark.torchscript_engine import TorchScriptEngine

from sparseml.pytorch.models.classification import resnet50
import shutil
from tests.utils.torch import find_pth_file_with_name
from deepsparse import Pipeline

TORCH_HUB = "~/.cache/torch"

@pytest.fixture(autouse=True, scope="module")
def delete_cached_torch_models():
    yield
    cache_dir = os.path.expanduser("~/.cache/torch")
    shutil.rmtree(cache_dir)
    assert os.path.exists(cache_dir) is False


def test_cpu_torchscript(torchvision_model_fixture):
    model = torchvision_model_fixture(pretrained=True)
    engine = TorchScriptEngine(model, device="cpu") 
    inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
    out = engine(inp)
    assert isinstance(out, List) and all(isinstance(arr, numpy.ndarray) for arr in out)

def test_cpu_torchscript_pipeline(torchvision_model_fixture):
    model = torchvision_model_fixture(pretrained=True)
    torch_model_path = os.path.join(TORCH_HUB, "hub", "checkpoints")
    model_path = find_pth_file_with_name(folder_path=torch_model_path, model_name="resnet50")
    torchscript_pipeline = Pipeline.create(
        task="image_classification",
        model_path=model_path,
        engine_type="torchscript",
        # task="example_task",
    )
    inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]

    pipeline_outputs = torchscript_pipeline(inp)
    # breakpoint()







# add test for pipeline
"""
download .pt model from the zoo into a temp folder
run test, 
delete when done

try:
    except:
        fianlly:
            shutil.rmtree(file_path)
"""
