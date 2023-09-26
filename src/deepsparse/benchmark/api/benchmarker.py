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


from typing import Optional

from deepsparse.benchmark.api.errors import UnclearBenchmarkerModeException
from deepsparse.benchmark.benchmark_model import benchmark_model
from deepsparse.benchmark.benchmark_pipeline import benchmark_pipeline


def run_benchmarker(
    model: Optional[str] = None,
    pipeline: Optional[str] = None,
    **kwargs,
):
    if bool(model) ^ bool(pipeline):
        if model:
            benchmarker = Benchmarker(model=model)
        elif pipeline:
            benchmarker = Benchmarker(pipeline=pipeline)

        return benchmarker(**kwargs)
    raise UnclearBenchmarkerModeException(
        "Benchmarker only accepts"
        "one input arg for "
        "'model' to run deepsparse.benchmark"
        "'pipeline' to run deepsparse.benchmark_pipeline"
    )


def _validate_exactly_one_mode_selected(
    *args,
):
    selections = sum(1 for mode in args if mode is not None)
    if selections != 1:
        raise UnclearBenchmarkerModeException(
            "Benchmarker only accepts"
            "one input arg for "
            "'model' to run deepsparse.benchmark"
            "'pipeline' to run deepsparse.benchmark_pipeline"
        )


class Benchmarker:
    """
    Benchmark API

    Input arg to `model`, `pipeline` should be one of:
     - SparseZoo stub
     - path to a model.onnx
     - path to a local folder containing a model.onnx
     - path to onnx.ModelProto

    Provide the stub/path to one of
     - onnx model to run deesparse.benchmark
     - deployment directory to run deepsparse deepsparse.benchmark_pipeline
    """

    def __init__(
        self,
        model: Optional[str] = None,
        pipeline: Optional[str] = None,
    ):
        _validate_exactly_one_mode_selected(model, pipeline)
        self.model = model
        self.pipeline = pipeline

    def __call__(self, **kwargs):
        if self.model:
            return benchmark_model(model_path=self.model, **kwargs)

        if self.pipeline:
            return benchmark_pipeline(model_path=self.pipeline, **kwargs)
