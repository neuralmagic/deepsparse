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
from deepsparse import analyze_model, compile_model
from deepsparse.utils import verify_outputs
from sparsezoo.models import classification
from sparsezoo.objects import Model


model_test_registry = {
    "mobilenet_v1": classification.mobilenet_v1,
    "mobilenet_v2": classification.mobilenet_v2,
    "resnet_18": classification.resnet_18,
    "efficientnet_b0": classification.efficientnet_b0,
}


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                model,
                b,
            )
            for model_name, model in model_test_registry.items()
            for b in [1, 16, 64]
        ]
    ),
)
class TestEngineParametrized:
    def test_engine(self, model: Model, batch_size: int):
        """
        Test the Engine.inference interfaces
        """
        m = model()
        batch = m.sample_batch(batch_size=batch_size)
        inputs = batch["inputs"]
        outputs = batch["outputs"]

        print("compile model")
        engine = compile_model(m, batch_size)

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

    def test_benchmark(self, model: Model, batch_size: int):
        """
        Test the Engine.benchmark() interface
        """

        m = model()
        batch = m.sample_batch(batch_size=batch_size)
        inputs = batch["inputs"]
        outputs = batch["outputs"]

        engine = compile_model(m, batch_size)
        results = engine.benchmark(inputs, include_outputs=True)

        for output in results.outputs:
            verify_outputs(output, outputs)

    def test_analyze(self, model: Model, batch_size: int):

        model = model()
        inputs = model.data_inputs.sample_batch(batch_size=batch_size)

        results = analyze_model(model, inputs, batch_size)
        print(results)
