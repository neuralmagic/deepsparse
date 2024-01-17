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

from deepsparse import TextGeneration
from deepsparse.transformers.pipelines.text_generation.pipeline import (
    TextGenerationPipeline,
)
from deepsparse.transformers.pipelines.text_generation.pipeline_no_kv_cache import (
    TextGenerationPipelineNoCache,
)


def test_create_pipeline_with_kv_cache_support(model_attributes):
    pipeline = TextGeneration(model_path=model_attributes[1])
    assert isinstance(pipeline, TextGenerationPipeline)
    pipeline = TextGeneration(model_path=model_attributes[1], onnx_model_name=None)
    assert isinstance(pipeline, TextGenerationPipeline)


def test_create_pipeline_with_no_kv_cache_support(model_attributes):
    pipeline = TextGeneration(
        model_path=model_attributes[1], onnx_model_name="model-orig.onnx"
    )
    assert isinstance(pipeline, TextGenerationPipelineNoCache)
