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

from deepsparse.benchmark.benchmark_model import benchmark_model
from deepsparse.benchmark.benchmark_pipeline import benchmark_pipeline


class Benchmarker:
    def __init__(self, model, pipeline, url):
        self.model = model
        self.pipeline = pipeline
        self.url = url

    def __call__(self, **kwargs):

        if self.model:
            benchmark_model(model_path=self.model, **kwargs)

        if self.pipeline:
            benchmark_pipeline(model_path=self.pipeline, **kwargs)

        if self.url:
            # benchmark with url here
            pass

    @staticmethod
    def benchmark(
        model: Optional[str] = None,
        pipeline: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,
    ):
        if len((model, pipeline, url)) != 1:
            return ValueError("[api.benchmark] Only one input arg required")

        if model:
            benchmarker = Benchmarker(model=model)
        elif pipeline:
            benchmarker = Benchmarker(pipeline=pipeline)
        elif url:
            benchmarker = Benchmarker(url=url)

        return benchmarker(**kwargs)
