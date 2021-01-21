import pytest
from inspect import getmembers, isfunction

from deepsparse import compile_model, analyze_model
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