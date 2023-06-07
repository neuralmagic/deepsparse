import pytest
from deepsparse import MetricCategories, Pipeline, TimingLogger
from tests.utils import mock_engine

@pytest.mark.parametrize(
    "pipeline_name",
    [
        ("python_logger"),
        (None),
    ],
)
@mock_engine(rng_seed=0)
def test_python_logger(engine, pipeline_name, capsys):
    pipeline = Pipeline.create(
            "token_classification",
            batch_size=1,
            logger=TimingLogger(),
        )
    for _ in range(2):
        pipeline("all_your_base_are_belong_to_us")
    pass
