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
from unittest import mock

import pytest
from deepsparse.benchmark.api.benchmarker import Benchmarker, run_benchmarker
from deepsparse.benchmark.api.errors import UnclearBenchmarkerModeException
from deepsparse.benchmark.config import PipelineBenchmarkConfig
from sparsezoo import Model


IC_STRING = "image_classification"
TEXT_GEN_STRING = "text_generation"

BENCHMARK_PIPELINE_IC_CONFIG_DICT = {
    "data_type": "dummy",
    "gen_sequence_length": 100,
    "input_image_shape": [500, 500, 3],
    "pipeline_kwargs": {},
    "input_schema_kwargs": {},
}

BENCHMARK_PIPELINE_TEXT_GEN_CONFIG_DICT = {
    "data_type": "dummy",
    "gen_sequence_length": 100,
    "pipeline_kwargs": {},
    "input_schema_kwargs": {},
}


class MockBenchmarker(Benchmarker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, **kwargs):
        if self.model:
            return "foo"

        if self.pipeline:
            pipeline_kwargs = kwargs["config"].__dict__
            if kwargs["task"] == IC_STRING:
                assert set(BENCHMARK_PIPELINE_IC_CONFIG_DICT).issubset(
                    set(pipeline_kwargs)
                )
            elif kwargs["task"] == TEXT_GEN_STRING:
                assert set(BENCHMARK_PIPELINE_TEXT_GEN_CONFIG_DICT).issubset(
                    set(pipeline_kwargs)
                )
            return "bar"


# @pytest.mark.skip(reason="Heavy load -- download text-gen models -- for GHA machine")
class TestBenchmarker:
    @pytest.fixture
    def get_model_path(self):
        """download model, return its path and delete at the end"""

        def download_model_and_return_path(
            stub: str, download_path: Optional[str] = None
        ):
            model = Model(stub, download_path)
            path = model.path
            yield path

            shutil.rmtree(path)
            assert os.path.exists(path) is False

        return download_model_and_return_path

    @pytest.fixture
    def benchmarker_fixture(self, get_model_path):
        def get(
            stub: str,
            task: Optional[str] = None,
            config_dict: Optional[str] = None,
            model_path: Optional[str] = None,
            model_args: Optional[Dict[str, Any]] = None,
            pipeline_args: Optional[Dict[str, Any]] = None,
        ):
            model_path = model_path or next(get_model_path(stub=stub))

            required_benchmark_model_args = model_args or {}

            required_benchmark_pipeline_args = pipeline_args or {
                "task": task,
                "config": PipelineBenchmarkConfig(**config_dict)
                if config_dict
                else None,
            }

            return (
                model_path,
                required_benchmark_model_args,
                required_benchmark_pipeline_args,
            )

        return get

    def test_validate_exactly_one_mode_selected(self):
        kwargs = {
            "model": "foo",
            "pipeline": "bar",
        }
        with pytest.raises(UnclearBenchmarkerModeException):
            Benchmarker(**kwargs)

    @pytest.mark.parametrize(
        "stub",
        [
            "zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/imagenet/base-none",
            (
                "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/"
                "bigpython_bigquery_thepile/base_quant-none"
            ),
        ],
    )
    def test_benchmark_model_from_benchmarker(self, benchmarker_fixture, stub):
        path, model_args, _ = benchmarker_fixture(stub=stub)
        benchmarker = Benchmarker(model=path)
        export_dict = benchmarker(**model_args)
        assert export_dict is not None

    @pytest.mark.parametrize(
        "stub,task,config_dict",
        [
            (
                (
                    "zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/"
                    "imagenet/base-none"
                ),
                IC_STRING,
                BENCHMARK_PIPELINE_IC_CONFIG_DICT,
            ),
            (
                (
                    "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/"
                    "bigpython_bigquery_thepile/base_quant-none"
                ),
                TEXT_GEN_STRING,
                BENCHMARK_PIPELINE_TEXT_GEN_CONFIG_DICT,
            ),
        ],
    )
    def test_benchmark_pipeline_from_benchmarker(
        self, benchmarker_fixture, stub, task, config_dict
    ):

        path, _, pipeline_args = benchmarker_fixture(
            stub=stub, task=task, config_dict=config_dict
        )
        # [TODO]: accept path for text_gen downstream benchmark_pipeline
        #  Passes for ic
        benchmarker = Benchmarker(pipeline=stub)

        batch_times, total_run_time, num_streams = benchmarker(**pipeline_args)
        assert batch_times is not None
        assert total_run_time is not None
        assert num_streams is not None

    @pytest.mark.parametrize(
        "stub,task,config_dict",
        [
            (
                (
                    "zoo:cv/classification/resnet_v1-50_2x/pytorch/sparseml/"
                    "imagenet/base-none"
                ),
                IC_STRING,
                BENCHMARK_PIPELINE_IC_CONFIG_DICT,
            ),
            (
                "zoo:nlg/text_generation/codegen_mono-350m/pytorch/huggingface/"
                "bigpython_bigquery_thepile/base_quant-none",
                TEXT_GEN_STRING,
                BENCHMARK_PIPELINE_TEXT_GEN_CONFIG_DICT,
            ),
        ],
    )
    def test_run_benchmarker__success(
        self,
        benchmarker_fixture,
        stub,
        task,
        config_dict,
    ):
        path, model_args, pipeline_args = benchmarker_fixture(
            stub=stub, task=task, config_dict=config_dict
        )

        with mock.patch(
            "deepsparse.benchmark.api.benchmarker.Benchmarker",
            side_effect=MockBenchmarker,
        ):
            response_model = run_benchmarker(model=path, **model_args)
            assert response_model == "foo"

            response_pipeline = run_benchmarker(pipeline=stub, **pipeline_args)
            assert response_pipeline == "bar"

    def test_run_benchmarker__failure(
        self,
    ):
        kwargs = {
            "model": "foo",
            "pipeline": "bar",
        }
        with pytest.raises(UnclearBenchmarkerModeException):
            run_benchmarker(**kwargs)
