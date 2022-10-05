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
from deepsparse.loggers.monitoring.monitoring import convert_data_to_estimates
from deepsparse.pipeline import Pipeline


CONFIG = {
    "pipeline_inputs": {"min_estimator": {"axis": 0}},
    "engine_inputs": {},
    "pipeline_outputs": {},
}

def _test_pipeline_inputs(pipeline_inputs, config):
    estimates = convert_data_to_estimates(pipeline_inputs, config)


@pytest.mark.parametrize(
    "pipeline_name, input, batch_size, config",
    [
        ("image_classification", {'images' : [numpy.random.normal(loc=0.0, scale = 1.0, size = (224, 224, 3))]}, 1, CONFIG)
    ],
)
def test_convert_data_to_estimates(pipeline_name, input, batch_size, config):
    pipeline = Pipeline.create(pipeline_name, batch_size=batch_size)
    pipeline_outputs, pipeline_inputs, engine_inputs, _ = pipeline.run_with_monitoring(**input)
    _test_pipeline_inputs(pipeline_inputs = pipeline_inputs, config = config["pipeline_inputs"])




# pipeline = Pipeline.create("image_classification", batch_size=1)
# pipeline_outputs, pipeline_inputs, engine_inputs, _ = pipeline.run_with_monitoring(
#     images=[numpy.random.rand(224, 224, 3)]
# )
# # # pipeline = Pipeline.create("yolact", batch_size=1)
# # # pipeline_outputs, pipeline_inputs, engine_inputs, _ = pipeline.run_with_monitoring(images = [numpy.random.rand(550, 550, 3)])
# # # pipeline = Pipeline.create("yolo", batch_size=1)
# # # pipeline_outputs, pipeline_inputs, engine_inputs, _ = pipeline.run_with_monitoring(images = [numpy.random.rand(550, 550, 3)])
# # pipeline = Pipeline.create("qa", batch_size=1)
# # pipeline_outputs, pipeline_inputs, engine_inputs, _ = pipeline.run_with_monitoring(question="What's my name?", context="My name is Snorlax")
# data_to_log = convert_data_to_estimates(pipeline_inputs, config["pipeline_inputs"])
# pass
