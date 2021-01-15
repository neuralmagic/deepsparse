import pytest
from deepsparse import analyze_model
from sparsezoo import Model
from sparsezoo.models.classification import mobilenet_v1


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
def test_analyze(params, model: Model, batch_size: int):

    model = model()
    batch = model.sample_batch(batch_size)
    inputs = batch["inputs"]

    results = analyze_model(model, inputs, batch_size)
