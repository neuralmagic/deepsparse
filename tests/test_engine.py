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

import pytest
from deepsparse import Engine, compile_model
from deepsparse.utils import verify_outputs
from sparsezoo import Model


model_test_registry = {
    "mobilenet_v1": (
        "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
    ),
    "mobilenet_v2": (
        "zoo:cv/classification/mobilenet_v2-1.0/pytorch/sparseml/imagenet/base-none"
    ),
    "resnet_18": (
        "zoo:cv/classification/resnet_v1-18/pytorch/sparseml/imagenet/base-none"
    ),
    "efficientnet_b0": (
        "zoo:cv/classification/efficientnet-b0/pytorch/sparseml/imagenet/base-none"
    ),
}


@pytest.mark.parametrize("batch_size", [1, 4, 16], scope="class")
@pytest.mark.parametrize("zoo_stub", model_test_registry.values(), scope="class")
@pytest.mark.smoke
class TestEngineParametrized:
    @pytest.fixture(scope="class")
    def model(self, zoo_stub: str):
        yield Model(zoo_stub)

    @pytest.fixture(scope="class")
    def engine(self, model: Model, batch_size: int):
        print("compile model")
        yield compile_model(model, batch_size, num_cores=1, num_streams=1)

    @pytest.fixture(scope="class")
    def engine_io(self, model: Model, batch_size: int):
        batch = model.sample_batch(batch_size=batch_size)
        input_key = next(key for key in batch.keys() if "input" in key)
        output_key = next(key for key in batch.keys() if "output" in key)
        yield batch[input_key], batch[output_key]

    def test_engine(self, engine: Engine, engine_io):
        """
        Test the Engine.inference interfaces
        """

        inputs, outputs = engine_io

        print("engine callable")
        pred_outputs = engine(inputs)
        verify_outputs(pred_outputs, outputs)

        print("engine run")
        pred_outputs = engine.run(inputs)
        verify_outputs(pred_outputs, outputs)

        print("engine mapped_run")
        pred_outputs = engine.mapped_run(inputs)
        assert len(pred_outputs) == len(outputs)

        print("engine timed_run")
        pred_outputs, elapsed = engine.timed_run(inputs)
        verify_outputs(pred_outputs, outputs)

    def test_benchmark(self, engine: Engine, engine_io):
        """
        Test the Engine.benchmark() interface
        """
        inputs, outputs = engine_io

        results = engine.benchmark(
            inputs, include_outputs=True, num_iterations=1, num_warmup_iterations=0
        )

        for output in results.outputs:
            verify_outputs(output, outputs)

    def test_analyze(self, engine: Engine, engine_io):
        inputs, _ = engine_io
        results = engine.analyze(inputs, num_iterations=1, num_warmup_iterations=0)
        assert "layer_info" in results
