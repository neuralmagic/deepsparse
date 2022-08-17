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

from concurrent.futures import ThreadPoolExecutor

import numpy
import os

import pytest
from deepsparse import Pipeline, Context

from .data_helpers import create_test_inputs


_SUPPORTED_TASKS = [
    # "text_classification",
    # "token_classification",
    # "yolo",
    # "image_classification",
    "yolact",
]

_BATCH_SIZES = [
    2,
]


def compare(expected, actual):
    assert type(expected) == type(actual)

    if isinstance(expected, (list, float, numpy.ndarray)):
        expected_np = numpy.asarray(expected, dtype=float)
        actual_np = numpy.asarray(actual, dtype=float)
        assert numpy.allclose(expected_np, actual_np, rtol=1e-3)
    elif isinstance(expected, dict):
        assert list(expected.keys()).sort() == list(actual.keys()).sort()
        for key, exp_value in expected.items():
            assert compare(exp_value, actual[key])
    else:
        assert expected == actual
    return True


@pytest.mark.parametrize("task", _SUPPORTED_TASKS)
def test_dynamic_is_same_as_static(task):
    os.environ["NM_BIND_THREADS_TO_CORES"] = "1"
    os.environ["NM_LOGGING_LEVEL"] = "diagnose"

    print(task)
    executor = ThreadPoolExecutor(max_workers=1)

    context = Context()

    # NOTE: re-use the same dynamic pipeline for different batch sizes
    print("creating dynamic pipeline")
    dynamic_pipeline = Pipeline.create(
        task=task, batch_size=None, executor=executor, context=context
    )
    print("creating dynamic pipeline done")
    assert dynamic_pipeline.use_dynamic_batch()

    for batch_size in _BATCH_SIZES:
        print("creating static pipeline")
        # NOTE: recompile model for each different batch_size
        static_pipeline = Pipeline.create(
            task=task, batch_size=batch_size, context=context
        )
        print("creating static pipeline done")
        assert not static_pipeline.use_dynamic_batch()

        print("creating test inputs")
        inputs = create_test_inputs(task=task, batch_size=batch_size)

        # run same outputs through both pipelines
        print("running dynamic pipeline")
        dynamic_outputs = dynamic_pipeline(**inputs)
        print("running static pipeline")
        static_outputs = static_pipeline(**inputs)
        print("done")

        assert isinstance(dynamic_outputs, dynamic_pipeline.output_schema)
        assert isinstance(static_outputs, static_pipeline.output_schema)

        expected_dict = static_outputs.dict()
        actual_dict = dynamic_outputs.dict()

        # Check that order is maintained
        try:
            assert static_outputs == dynamic_outputs
        except Exception:
            assert compare(expected_dict, actual_dict)

    executor.shutdown(wait=False)
    del os.environ["NM_LOGGING_LEVEL"]
