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

import os
import shutil
from typing import Any, Dict, Optional

import pytest
from deepsparse.benchmark.api.benchmarker import Benchmarker
from deepsparse.benchmark.api.errors import UnclearBenchmarkerModeException
from sparsezoo import Model


@pytest.fixture(scope="function")
def get_model_path():
    """download model, return its path and delete at the end"""

    text_gen_stub = "zoo:opt-1.3b-opt_pretrain-quantW8A8"
    default_download_path = os.path.expanduser(
        os.path.join("~/.cache/nm_tests", "deepsparse")
    )

    def download_model_and_return_path(
        stub: Optional[str] = None, download_path: Optional[str] = None
    ):
        model = Model(stub or text_gen_stub, download_path or default_download_path)
        yield model.path

        # yield model.path()
        # shutil.rmtree(path)
        # assert os.path.exists(path) is False

    return download_model_and_return_path


@pytest.fixture
def benchmarker_fixture(get_model_path):
    def get(
        source: Optional[str] = None,
        path: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        pipeline_args: Dict[str, Any] = None,
    ):
        model_path = path or next(get_model_path(stub=source))

        required_benchmark_model_args = model_args or {}

        required_benchmark_pipeline_args = pipeline_args or {
            "task": "text_generation",
            "config": "",
        }

        return (
            model_path,
            required_benchmark_model_args,
            required_benchmark_pipeline_args,
        )

    return get

    # required_benchmark_pipeline_args = {
    #     "task": "text_generation",
    # }

    # return model_path, required_benchmark_pipeline_args


# def test_validate_exactly_one_arg_provided():
#     args = {
#         "model": "foo",
#         "pipeline": "bar",
#     }
#     with pytest.raises(UnclearBenchmarkerModeException):
#         Benchmarker(**args)


# def test_benchmark_model_from_benchmarker(benchmarker_fixture):
#     path, model_args, _ = benchmarker_fixture()
#     benchmarker = Benchmarker(model=path)
#     export_dict = benchmarker(**model_args)
# assert export_dict is not None


def test_benchmark_pipeline_from_benchmarker(benchmarker_fixture):
    path, _, pipeline_args = benchmarker_fixture()
    benchmarker = Benchmarker(pipeline=path)
    batch_times, total_run_time, num_streams = benchmarker(**pipeline_args)
    assert batch_times is not None
    assert total_run_time is not None
    assert num_streams is not None
