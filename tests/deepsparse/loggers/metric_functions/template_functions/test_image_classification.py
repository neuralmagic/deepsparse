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
    python:
data_logging:
    - func: image_classification
      frequency: 2"""


@pytest.mark.parametrize(
    "config, inp, num_iterations",
    [
        (yaml_config, [numpy.ones((3, 224, 224))] * 2, 6),
        (yaml_config, numpy.ones((2, 3, 224, 224)), 6),
    ],
)
@mock_engine(rng_seed=0)
def test_end_to_end(mock_engine, config, inp, num_iterations):
    pipeline = Pipeline.create("image_classification", logger=config)
    for _ in range(num_iterations):
        pipeline(images=inp)
