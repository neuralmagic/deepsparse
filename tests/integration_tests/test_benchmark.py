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
def test_benchmark(model: Model, batch_size: int):

    m = model()
    batch = m.sample_batch(batch_size=batch_size)
    inputs = batch["inputs"]
    outputs = batch["outputs"]

    engine = compile_model(m, batch_size)
    results = engine.benchmark(inputs, include_outputs=True)

    for output in results.outputs:
        verify_outputs(output, outputs)
