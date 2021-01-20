import pytest
from deepsparse import analyze_model
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
def test_analyze(model: Model, batch_size: int):

    model = model()
    inputs = model.data_inputs.sample_batch(batch_size=batch_size)

    results = analyze_model(model, inputs, batch_size)
    print(results)
