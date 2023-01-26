import pytest
import numpy
from tests.utils import mock_engine
from deepsparse import Pipeline

yaml_config = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    predefined:
    - func: object_detection
      frequency: 1"""

expected_logs = """identifier:yolo/pipeline_inputs.images__image_shape, value:{'channels': 3, 'dim_0': 640, 'dim_1': 640}, category:MetricCategories.DATA
identifier:yolo/pipeline_inputs.images__mean_pixels_per_channel, value:{'channel_0': 1.0, 'channel_1': 1.0, 'channel_2': 1.0}, category:MetricCategories.DATA
identifier:yolo/pipeline_inputs.images__std_pixels_per_channel, value:{'channel_0': 0.0, 'channel_1': 0.0, 'channel_2': 0.0}, category:MetricCategories.DATA
identifier:yolo/pipeline_inputs.images__fraction_zeros, value:0.0, category:MetricCategories.DATA
identifier:yolo/pipeline_outputs.labels__detected_classes, value:{'19.0': 8, '79.0': 10, '27.0': 10, '77.0': 8, '55.0': 8, '45.0': 8, '10.0': 14, '20.0': 6, '39.0': 8, '1.0': 10, '54.0': 12, '22.0': 10, '29.0': 12, '73.0': 8, '3.0': 8, '40.0': 12, '44.0': 8, '59.0': 8, '67.0': 12, '63.0': 12, '64.0': 8, '62.0': 14, '47.0': 14, '71.0': 8, '32.0': 4, '13.0': 8, '68.0': 8, '60.0': 6, '26.0': 8, '38.0': 8, '36.0': 8, '4.0': 12, '76.0': 8, '70.0': 8, '31.0': 8, '58.0': 8, '53.0': 8, '51.0': 10, '9.0': 4, '37.0': 2, '66.0': 6, '35.0': 6, '5.0': 2, '75.0': 8, '6.0': 6, '50.0': 10, '34.0': 8, '49.0': 6, '8.0': 8, '56.0': 10, '52.0': 6, '57.0': 6, '30.0': 6, '46.0': 6, '72.0': 8, '0.0': 4, '25.0': 8, '33.0': 10, '69.0': 8, '16.0': 6, '7.0': 6, '15.0': 10, '14.0': 10, '21.0': 6, '17.0': 6, '43.0': 8, '48.0': 6, '42.0': 8, '78.0': 2, '11.0': 4, '18.0': 10, '65.0': 4, '12.0': 8, '61.0': 2, '41.0': 4, '2.0': 2, '24.0': 6, '28.0': 4, '23.0': 2}, category:MetricCategories.DATA
identifier:yolo/pipeline_outputs.labels__number_detected_objects, value:[300, 300], category:MetricCategories.DATA
identifier:yolo/pipeline_outputs.scores__mean_score_per_detection, value:[0.9865101563930512, 0.9865101563930512], category:MetricCategories.DATA
identifier:yolo/pipeline_outputs.scores__std_score_per_detection, value:[0.00563188730342948, 0.00563188730342948], category:MetricCategories.DATA
"""  # noqa E501

# TODO Why results so weird
@pytest.mark.parametrize(
    "config, inp, num_iterations, expected_logs",
    [
        (yaml_config, [numpy.ones((3, 640, 640))] * 2, 1, expected_logs),
        (yaml_config, numpy.ones((2, 3, 640, 640)), 1, expected_logs),
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config, inp, num_iterations, expected_logs):
    pipeline = Pipeline.create("yolo", logger=config)
    for _ in range(num_iterations):
        pipeline(images=inp)

    logs = pipeline.logger.loggers[0].logger.loggers[0].calls
    data_logging_logs = [log for log in logs if "DATA" in log]
    for log, expected_log in zip(data_logging_logs, expected_logs.splitlines()):
        assert log == expected_log