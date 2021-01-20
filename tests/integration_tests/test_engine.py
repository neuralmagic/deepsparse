import pytest
from deepsparse import compile_model
from deepsparse.utils import verify_outputs
from sparsezoo.models.classification import mobilenet_v1
from sparsezoo.objects import Model


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                mobilenet_v1,
                b,
            )
            for b in [1, 16, 64]
        ]
    ),
)
def test_engine(model: Model, batch_size: int):

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
