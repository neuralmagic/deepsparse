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

import numpy as np

import pytest
from deepsparse import Engine, model_debug_analysis
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
        yield Engine(model, batch_size)

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

        print("engine batched_run")
        # make some fake padded data
        stacked_inputs = [np.repeat(array, 3, axis=0) for array in inputs]
        stacked_outputs = [np.repeat(array, 3, axis=0) for array in outputs]
        pred_outputs = engine.batched_run(stacked_inputs)
        verify_outputs(pred_outputs, stacked_outputs)

        print("engine input_shapes")
        pred_input_shapes = engine.input_shapes
        assert pred_input_shapes[0] == inputs[0].shape

        print("engine output_shapes")
        pred_output_shapes = engine.output_shapes
        assert pred_output_shapes[0] == outputs[0].shape

        # Note: These are hardcoded for the model_test_registry models of
        # mobilenet_v1, mobilenet_v2, resnet_v1-18, efficientnet-b0
        print("engine input_names")
        assert "input" in engine.input_names[0]
        print("engine output_names")
        assert "output_0" in engine.output_names[0]
        assert "output_1" in engine.output_names[1]

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


@pytest.mark.parametrize("batch_size", [1, 16], scope="class")
@pytest.mark.parametrize("zoo_stub", model_test_registry.values(), scope="class")
@pytest.mark.smoke
class TestDebugAnalysisEngineParametrized:
    @pytest.fixture(scope="class")
    def model(self, zoo_stub: str):
        yield Model(zoo_stub)

    @pytest.fixture(scope="class")
    def engine_io(self, model: Model, batch_size: int):
        batch = model.sample_batch(batch_size=batch_size)
        input_key = next(key for key in batch.keys() if "input" in key)
        output_key = next(key for key in batch.keys() if "output" in key)
        yield batch[input_key], batch[output_key]

    def test_analyze(self, model: Model, batch_size: int, engine_io):
        inputs, _ = engine_io
        results = model_debug_analysis(
            model,
            inputs,
            batch_size,
            num_iterations=1,
            num_warmup_iterations=0,
        )
        assert "layer_info" in results


@pytest.mark.smoke
class TestBatchedEngine:
    def test_batched(self):
        model_stub = (
            "zoo:cv/classification/mobilenet_v1-1.0/pytorch/sparseml/imagenet/base-none"
        )

        # batch_size=None disable batch override
        engine = Engine(model_stub, batch_size=None, input_shapes=[3, 3, 224, 224])
        assert engine.input_shapes[0] == (3, 3, 224, 224)
        assert engine.generate_random_inputs()[0].shape == (3, 3, 224, 224)

        # Engine implicitly assumes batch size 1
        engine = Engine(model_stub)
        assert engine.input_shapes[0] == (1, 3, 224, 224)
        assert engine.generate_random_inputs()[0].shape == (1, 3, 224, 224)

        # Engine first applies input_shapes, then applies batch override to the model
        engine = Engine(model_stub, batch_size=5, input_shapes=[1, 3, 224, 224])
        assert engine.input_shapes[0] == (5, 3, 224, 224)
        assert engine.generate_random_inputs()[0].shape == (5, 3, 224, 224)
