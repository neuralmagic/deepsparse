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

import numpy

import pytest
from deepsparse import Pipeline
from tests.utils import mock_engine


yaml_config = """
loggers:
    list_logger:
        path: tests/deepsparse/loggers/helpers.py:ListLogger
data_logging:
    - func: image_classification
      frequency: 2"""

expected_logs = """identifier:image_classification/pipeline_inputs.images__image_shape__channels, value:3, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__image_shape__dim_0, value:224, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__image_shape__dim_1, value:224, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__mean_pixels_per_channel__channel_0, value:1.0, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__mean_pixels_per_channel__channel_1, value:1.0, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__mean_pixels_per_channel__channel_2, value:1.0, category:MetricCategories.DATA
identifier:image_classification/pipeline_inputs.images__fraction_zeros, value:0.0, category:MetricCategories.DATA"""  # noqa E501


@pytest.mark.parametrize(
    "config, inp, num_iterations, expected_logs",
    [
        (yaml_config, [numpy.ones((3, 224, 224))] * 2, 1, expected_logs),
        (yaml_config, numpy.ones((2, 3, 224, 224)), 1, expected_logs),
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config, inp, num_iterations, expected_logs):
    pipeline = Pipeline.create("image_classification", logger=config)
    for _ in range(num_iterations):
        pipeline(images=inp)
    logs = pipeline.logger.logger.loggers[0].logger.loggers[0].calls
    data_logging_logs = [log for log in logs if "DATA" in log]
    for log, expected_log in zip(data_logging_logs, expected_logs.splitlines()):
        assert log == expected_log
