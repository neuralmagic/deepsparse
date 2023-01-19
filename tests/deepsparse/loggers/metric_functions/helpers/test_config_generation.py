import pytest
import os
from deepsparse.loggers.metric_functions.helpers.config_generation import data_logging_config_from_predefined
from deepsparse.loggers.config import PipelineLoggingConfig

config_1 = """
loggers:
    python:
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 1
    ..."""

config_2 = """
loggers:
    python:
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 1
    ..."""

config_3 = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    pipeline_inputs.images:
        - func: image_shape
        - frequency: 2
    ..."""
@pytest.mark.parametrize(
    "group_names, loggers, frequency, save_dir, expected_config",
    [
        ("image_classification", None, None, None, config_1),
        (["image_classification"], None, None, None, config_1),
        (["image_classification", "image_segmentation"], None, None, None, config_2),
        (["image_classification"], {"list_logger": {"path": "tests/deepsparse/loggers/helpers.py:ListLogger"}, 2, "folder", config_3),
    ]
)
def test_data_logging_config_from_predefined(tmp, group_names, loggers, frequency, save_dir, expected_config):
    config = data_logging_config_from_predefined(group_names, loggers, frequency, save_dir)
    assert config == expected_config
    assert PipelineLoggingConfig(config)
    if save_dir:
        with open(os.path.join(tmp, save_dir, "data_logging_config.yaml")) as f:
            assert f.read() == expected_config