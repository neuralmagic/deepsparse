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

import pytest
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from deepsparse.evaluation.integrations.perplexity import (
    integration_eval,
    load_perplexity_dataset,
)

from evaluate import load


@pytest.mark.parametrize(
    "model_path, model_id",
    [
        (
            "hf:mgoin/TinyStories-1M-deepsparse",
            "roneneldan/TinyStories-1M",
        )
    ],
)
@pytest.mark.parametrize(
    "datasets",
    [
        "openai_humaneval",
        # TODO: add more datasets
        # "c4",
        # "wikitext2",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 3],
)
class TestLMEvaluationHarness:
    def test_perplexity_with_kv_cache(self, model_path, model_id, datasets, batch_size):
        limit = 5

        result_gt = self._get_ground_truth(
            datasets=datasets, batch_size=batch_size, limit=limit, model_id=model_id
        )

        result = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=limit,
        )

    def test_perplexity_with_no_kv_cache_pipeline(
        self, model_path, model_id, datasets, batch_size
    ):
        limit = 5

        result_gt = self._get_ground_truth(
            datasets=datasets, batch_size=batch_size, limit=limit, model_id=model_id
        )

        result = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
                onnx_model_name="model-orig.onnx",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=limit,
        )

    @staticmethod
    def _get_ground_truth(datasets, batch_size, limit, model_id):
        perplexity = load("perplexity", module_type="metric")
        dataset, *_ = load_perplexity_dataset(dataset_name=datasets, split="test")
        predictions = []
        for i, sample in enumerate(dataset):
            if i == batch_size * limit:
                break
            predictions.append(sample["prompt"] + sample["canonical_solution"])
        return perplexity.compute(predictions=predictions, model_id=model_id)
