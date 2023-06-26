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

import pytest
from deepsparse.benchmark.torch_engine import TorchEngine
from sparseml.pytorch.models.classification import resnet50


@pytest.mark.parametrize("device", [("cpu"), ("cuda")])
def test_torchscript(device):
    model = resnet50(pretrained=True)
    engine = TorchEngine(model, device=device)
    inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
    out = engine(inp)
    assert isinstance(out, List) and all(isinstance(arr, numpy.ndarray) for arr in out)
