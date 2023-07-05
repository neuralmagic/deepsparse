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

from deepsparse import Pipeline
from deepsparse.benchmark.torchscript_engine import TorchScriptEngine
from deepsparse.image_classification.schemas import ImageClassificationOutput


try:
    import torch

    torch_import_error = None
except Exception as torch_import_err:
    torch_import_error = torch_import_err
    torch = None


def test_cpu_torchscript(torchscript_test_setup):
    models = torchscript_test_setup
    for model in models.values():
        engine = TorchScriptEngine(model, device="cpu")
        inp = [numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)]
        out = engine(inp)
        assert isinstance(out, List) and all(
            isinstance(arr, numpy.ndarray) for arr in out
        )


def test_cpu_torchscript_pipeline(torchscript_test_setup):
    models = torchscript_test_setup

    torchscript_pipeline = Pipeline.create(
        task="image_classification",
        model_path=models["jit_model_path"],
        engine_type="torchscript",
        image_size=(224, 224),
    )

    inp = [numpy.random.rand(3, 224, 224).astype(numpy.float32)]
    pipeline_outputs = torchscript_pipeline(images=inp)
    assert isinstance(pipeline_outputs, ImageClassificationOutput)
