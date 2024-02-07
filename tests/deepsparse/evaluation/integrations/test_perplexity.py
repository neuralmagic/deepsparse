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

from copy import copy

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
        "wikitext2",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
class TestPerplexity:
    limit = 2

    def test_perplexity_ground_truth_equal_pipeline(
        self, model_path, model_id, datasets, batch_size
    ):
        # setting max_sequence_length to 16 to speed up the test
        kwargs_ground_truth = (
            dict(max_sequence_length=16) if datasets in {"c4", "wikitext2"} else {}
        )
        kwargs = copy(kwargs_ground_truth)

        result_gt = self._get_ground_truth(
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
            model_id=model_id,
            kwargs=kwargs_ground_truth,
        )

        result = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=self.limit,
            # we are setting accumulate=False to compare
            # with the torch ground truth apples to apples
            accumulate=False,
            **kwargs,
        )
        perplexities = result.formatted[0].metrics[0].value
        perplexities_gt = result_gt["perplexities"]
        assert np.allclose(perplexities, perplexities_gt, rtol=0.1)

    def test_perplexity_kv_cache_pipeline_equal_no_kv_cache_pipeline(
        self, model_path, model_id, datasets, batch_size
    ):

        kwargs_ground_truth = (
            dict(max_sequence_length=16) if datasets in {"c4", "wikitext2"} else {}
        )
        kwargs = copy(kwargs_ground_truth)

        result_kv_cache = integration_eval(
            pipeline=TextGenerationPipeline(
                model_path="hf:mgoin/TinyStories-1M-deepsparse",
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            model_path=model_id,
            batch_size=batch_size,
            limit=self.limit,
            **kwargs,
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
            **kwargs,
        )

        perplexities_kv_cache = result_kv_cache.formatted[0].metrics[0].value
        perplexities_non_kv_cache = result_non_kv_cache.formatted[0].metrics[0].value
        np.allclose(perplexities_kv_cache, perplexities_non_kv_cache, rtol=0.1)

    @staticmethod
    def _get_ground_truth(datasets, batch_size, limit, model_id, kwargs={}):
        perplexity = load("perplexity", module_type="metric")
        kwargs["model_path"] = model_id
        dataset, *_ = load_perplexity_dataset(dataset_name=datasets, **kwargs)
        predictions = []
        for i, sample in enumerate(dataset):
            if i == batch_size * limit:
                break
            predictions.append(sample)
        return perplexity.compute(
            predictions=predictions, add_start_token=False, model_id=model_id
        )
