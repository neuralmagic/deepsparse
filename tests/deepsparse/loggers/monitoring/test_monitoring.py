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

from typing import Any

import numpy
from pydantic import BaseModel

import pytest
from deepsparse.loggers import extract_logging_data, extract_numpy_data


class PipelineInputs(BaseModel):
    numpy_field: numpy.ndarray
    non_numpy_field: Any

    class Config:
        arbitrary_types_allowed = True


numpy_array = numpy.arange(-10, 10, 1).reshape(2, 10)
pipeline_inputs = PipelineInputs(numpy_field=numpy_array, non_numpy_field="some_string")
pipeline_inputs_numpy = {"numpy_field": numpy_array}


@pytest.mark.parametrize(
    "data_input, expected_result",
    [(pipeline_inputs, pipeline_inputs_numpy)],
)
def test_extract_numpy_data(data_input, expected_result):
    result = extract_numpy_data(data_input)
    assert result == expected_result


@pytest.mark.parametrize(
    "config, pipeline_inputs, engine_inputs, pipeline_outputs, "
    "expected_result, raise_warning, raise_error",
    [
        (
            {"pipeline_inputs": {"min_estimator": {"axis": 0}}},
            pipeline_inputs,
            None,
            None,
            {
                "pipeline_inputs": {"numpy_field": {"min_estimator_result": -10}},
                "pipeline_outputs": {},
                "engine_inputs": {},
            },
            False,
            False,
        ),
        (
            {"pipeline_inputs": {"min_estimator": {}}},
            pipeline_inputs,
            None,
            None,
            {
                "pipeline_inputs": {"numpy_field": {"min_estimator_result": -10}},
                "pipeline_outputs": {},
                "engine_inputs": {},
            },
            False,
            False,
        ),
        (
            {"pipeline_inputs": {"min_estimator": {}}},
            None,
            None,
            None,
            {"pipeline_inputs": {}, "pipeline_outputs": {}, "engine_inputs": {}},
            True,
            False,
        ),
        (
            {"pipeline_inputs": {"RAISE_ERROR_estimator": {}}},
            pipeline_inputs,
            None,
            None,
            {"pipeline_inputs": {}, "pipeline_outputs": {}, "engine_inputs": {}},
            False,
            True,
        ),
    ],
)
def test_extract_logging_data(
    config,
    pipeline_inputs,
    engine_inputs,
    pipeline_outputs,
    expected_result,
    raise_warning,
    raise_error,
):
    if raise_error:
        with pytest.raises(AttributeError):
            extract_logging_data(
                logging_config=config,
                pipeline_inputs=pipeline_inputs,
                engine_inputs=engine_inputs,
                pipeline_outputs=pipeline_outputs,
            )
    elif raise_warning:
        with pytest.warns(UserWarning):
            extract_logging_data(
                logging_config=config,
                pipeline_inputs=pipeline_inputs,
                engine_inputs=engine_inputs,
                pipeline_outputs=pipeline_outputs,
            )

    else:
        result = extract_logging_data(
            logging_config=config,
            pipeline_inputs=pipeline_inputs,
            engine_inputs=engine_inputs,
            pipeline_outputs=pipeline_outputs,
        )

        assert result == expected_result
