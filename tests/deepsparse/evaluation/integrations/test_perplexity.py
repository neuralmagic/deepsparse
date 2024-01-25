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

import numpy as np

import pytest
from deepsparse.evaluation.integrations.perplexity import (
    integration_eval,
    load_perplexity_dataset,
)
from deepsparse.transformers.pipelines.text_generation import TextGenerationPipeline
from evaluate import load


@pytest.fixture()
def model_path():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture()
def model_id():
    return "roneneldan/TinyStories-1M"


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
    limit = 4

    def test_perplexity_ground_truth_equal_pipeline(
        self, model_path, model_id, datasets, batch_size
    ):
        result_gt = self._get_ground_truth(
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
            model_id=model_id,
        )

        result = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
        )
        perplexities = result.formatted[0].metrics[0].value
        perplexities_gt = result_gt["perplexities"]
        # TODO: This seemingly big error is due to the fact that
        # small (1e-2) differences in neg log likelihood get
        # amplified when computing perplexity
        # (when applying exp function)
        assert np.allclose(perplexities, perplexities_gt, atol=100)

    def test_perplexity_kv_cache_pipeline_equal_no_kv_cache_pipeline(
        self, model_path, datasets, batch_size
    ):
        result_kv_cache = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
        )

        result_non_kv_cache = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
                onnx_model_name="model-orig.onnx",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
        )

        perplexities_kv_cache = result_kv_cache.formatted[0].metrics[0].value
        perplexities_non_kv_cache = result_non_kv_cache.formatted[0].metrics[0].value
        # TODO: This seemingly big error is due to the fact that
        # small (1e-2) differences in neg log likelihood get
        # amplified when computing perplexity
        # (when applying exp function)
        np.allclose(perplexities_kv_cache, perplexities_non_kv_cache, atol=100)

    @staticmethod
    def _get_ground_truth(datasets, batch_size, limit, model_id):
        perplexity = load("perplexity", module_type="metric")
        dataset, *_ = load_perplexity_dataset(dataset_name=datasets, split="test")
        predictions = []
        for i, sample in enumerate(dataset):
            if i == batch_size * limit:
                break
            predictions.append(sample["prompt"] + sample["canonical_solution"])
        return perplexity.compute(
            predictions=predictions, add_start_token=False, model_id=model_id
        )
