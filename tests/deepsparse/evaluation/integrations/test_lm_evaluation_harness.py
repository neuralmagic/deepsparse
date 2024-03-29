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
from deepsparse.evaluation.integrations import try_import_lm_evaluation_harness
from deepsparse.evaluation.utils import create_pipeline


@pytest.mark.parametrize(
    "batch_size",
    [1, 3],
)
@pytest.mark.skipif(
    not try_import_lm_evaluation_harness(raise_error=False),
    reason="lm_evaluation_harness not installed",
)
class TestLMEval:
    @pytest.fixture()
    def integration_eval(self):
        from deepsparse.evaluation.integrations.lm_evaluation_harness import (
            integration_eval as eval_fn,
        )

        return eval_fn

    @pytest.mark.parametrize(
        "datasets",
        [
            "hellaswag",
            ["arc_challenge"],
            ["hellaswag", "arc_challenge"],
        ],
    )
    def test_likelihood_scenario(self, batch_size, datasets, integration_eval):

        model_path_ds = "hf:mgoin/TinyStories-1M-ds"
        model_path_hf = "roneneldan/TinyStories-1M"
        limit = 2

        out_onnx = integration_eval(
            create_pipeline(
                model_path_ds,
                engine_type="onnxruntime",
            ),
            datasets=datasets,
            batch_size=batch_size,
            limit=limit,
            use_cache=None,  # avoid saving files when running tests
        )

        from lm_eval import evaluator, tasks, utils

        datasets_ = (",").join(datasets) if isinstance(datasets, list) else datasets
        out_torch = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path_hf}",
            tasks=utils.pattern_match(datasets_.split(","), tasks.ALL_TASKS),
            batch_size=batch_size,
            limit=limit,
            use_cache=None,  # avoid saving files when running tests
        )
        self._test_same(out_onnx.raw, out_torch, datasets)

    @pytest.mark.parametrize(
        "datasets",
        [
            "gsm8k",
        ],
    )
    def test_greedy_until_scenario(self, batch_size, datasets, integration_eval):
        model_path_ds = "hf:mgoin/TinyLlama-1.1B-step-50K-105b-ONNX"
        model_path_hf = "TinyLlama/TinyLlama-1.1B-step-50K-105b"
        limit = 2
        # compute until 16 new tokens
        # so that tests are faster
        gen_kwargs = "max_gen_toks=16"

        out_onnx = integration_eval(
            create_pipeline(model_path_ds, engine_type="onnxruntime"),
            datasets=datasets,
            batch_size=batch_size,
            limit=limit,
            gen_kwargs=gen_kwargs,
            use_cache=None,  # avoid saving files when running tests
        )

        from lm_eval import evaluator, tasks, utils

        datasets_ = (",").join(datasets) if isinstance(datasets, list) else datasets
        out_torch = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path_hf}",
            tasks=utils.pattern_match(datasets_.split(","), tasks.ALL_TASKS),
            batch_size=batch_size,
            limit=limit,
            gen_kwargs=gen_kwargs,
            use_cache=None,  # avoid saving files when running tests
        )
        self._test_same(out_onnx.raw, out_torch, datasets)

    @staticmethod
    def _test_same(out_onnx, out_torch, datasets, greedy=False):
        datasets = datasets if isinstance(datasets, list) else [datasets]
        for dataset in datasets:
            torch_samples = out_torch["samples"][dataset]
            onnx_samples = out_onnx["samples"][dataset]
            for torch_sample, onnx_sample in zip(torch_samples, onnx_samples):
                if greedy:
                    # for datasets that validate greedy generation
                    # make sure that generated sequences are the same
                    assert torch_sample["resps"] == onnx_sample["resps"]
                else:
                    # for datasets that validate likelihood
                    # make sure that likelihoods are the same
                    assert (
                        pytest.approx(torch_sample["resps"][0][0], 0.0001)
                        == onnx_sample["resps"][0][0]
                    )
