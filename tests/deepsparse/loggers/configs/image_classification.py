import pytest
from tests.utils import mock_engine
from deepsparse import Pipeline

yaml_config= """
loggers:
    python:
data_logging:
    __all__:
        - func: image_classification
          frequency: 3"""

@pytest.mark.parametrize(
    "config",
    [
        yaml_config
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config):
    Pipeline.create("image_classification", logger=config)