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

import numpy
import os
import pytest
from deepsparse.benchmark.torchscript_engine import TorchScriptEngine

import shutil
from tests.utils.torch import find_file_with_pattern, save_pth_to_pt
from deepsparse import Pipeline

from deepsparse.image_classification.schemas import (
    ImageClassificationOutput, 
)

import torch


# parameterize input as nn.Module or as jit
"""
four tests, 
1. nModule
2. jit
3. path to .pth
4. path to .pt
"""


TORCH_HUB = "~/.cache/torch"

# @pytest.fixture(scope="module")
@pytest.fixture(scope="function")
def torchscript_test_setup(torchvision_model_fixture):
    path = os.path.expanduser(os.path.join(TORCH_HUB, "hub", "checkpoints"))

    resnet50_nn_module = torchvision_model_fixture(pretrained=True, return_jit=False)
    expr = r"^resnet50-[0-9a-z]+\.pt[h]?$"
    resnet50_nn_module_path = find_file_with_pattern(path, expr)

    resnet50_jit = torchvision_model_fixture(pretrained=True, return_jit=True)
    resnet50_jit_path = resnet50_nn_module_path.replace(".pth", ".pt")
    torch.jit.save(resnet50_jit, resnet50_jit_path)
    # resnet50_jit.save(resnet50_jit_path)
    # torch.jit.save(m, '/home/ubuntu/george/nm/deepsparse/scriptmodule.pt')


    yield {
        "jit_model": resnet50_jit,
        "jit_model_path": resnet50_jit_path,
    }

    # cache_dir = os.path.expanduser("~/.cache/torch")
    # shutil.rmtree(cache_dir)
    # assert os.path.exists(cache_dir) is False


def test_cpu_torchscript(torchscript_test_setup):
    models = torchscript_test_setup
    for model in models.values():
        engine = TorchScriptEngine(model, device="cpu") 
        inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        out = engine(inp)
        assert isinstance(out, List) and all(isinstance(arr, numpy.ndarray) for arr in out)


def test_cpu_torchscript_pipeline(torchscript_test_setup):
    models = torchscript_test_setup

    torchscript_pipeline = Pipeline.create(
        task="image_classification",
        # model_path=models["jit_model_path"],
        # model_path='/home/ubuntu/george/nm/deepsparse/scriptmodule.pt',
        model_path='/home/ubuntu/george/nm/deepsparse/traced_resnet_model.pt',
        # model_path='/home/ubuntu/george/nm/deepsparse/resnet50.pt',


        engine_type="torchscript",
        image_size = (224, 224),
    )

    inp = [numpy.random.rand(3, 224, 224).astype(numpy.float32)]
    # breakpoint()
    pipeline_outputs = torchscript_pipeline(images=inp)
    assert isinstance(pipeline_outputs, ImageClassificationOutput)


# def test_cpu_torchscript_pipeline(torchvision_model_fixture):
#     model = torchvision_model_fixture(pretrained=True)
#     torch_model_path = os.path.join(TORCH_HUB, "hub", "checkpoints")
#     model_path_pth = find_pth_file_with_name(folder_path=torch_model_path, model_name="resnet50")
#     model_path_pt = save_pth_to_pt(model_path_pth)
#     # breakpoint()
#     model_path_pt = '/home/ubuntu/george/nm/deepsparse/resnet50.pt'

#     torchscript_pipeline = Pipeline.create(
#         task="image_classification",
#         model_path=model_path_pt,
#         engine_type="torchscript",
#         image_size = (224, 224),
#     )

#     inp = [numpy.random.rand(3, 224, 224).astype(numpy.float32)]
#     breakpoint()
#     pipeline_outputs = torchscript_pipeline(images=inp)
#     assert isintance(pipeline_outputs, ImageClassificationOutput)

