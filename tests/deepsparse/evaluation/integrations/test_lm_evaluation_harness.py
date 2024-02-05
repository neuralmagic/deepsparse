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
from deepsparse.evaluation.utils import create_model_from_target


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
        "datasets_likelihood",
        [
            "hellaswag",
            ["arc_challenge"],
            ["hellaswag", "arc_challenge"],
        ],
    )
    def test_likelihood_scenario(
        self, batch_size, datasets_likelihood, integration_eval
    ):
        model_path_ds = "hf:mgoin/TinyStories-1M-ds"
        model_path_hf = "roneneldan/TinyStories-1M"

        out_onnx = integration_eval(
            model=create_model_from_target(
                model_path_ds,
                engine_type="onnxruntime",
            ),
            datasets=datasets_likelihood,
            batch_size=batch_size,
            limit=2,
            use_cache=None,  # avoid saving files when running tests
        )

        out_torch = integration_eval(
            model="hf",
            model_args=f"pretrained={model_path_hf}",
            datasets=datasets_likelihood,
            batch_size=batch_size,
            limit=2,
            use_cache=None,  # avoid saving files when running tests
        )
        self._test_same(out_onnx, out_torch, datasets_likelihood)

    @pytest.mark.parametrize(
        "datasets_greedy_until",
        [
            "gsm8k",
        ],
    )
    def test_greedy_until_scenario(
        self, batch_size, datasets_greedy_until, integration_eval
    ):
        model_path_ds = "hf:mgoin/TinyLlama-1.1B-step-50K-105b-ONNX"
        model_path_hf = "TinyLlama/TinyLlama-1.1B-step-50K-105b"

        out_onnx = integration_eval(
            model=create_model_from_target(model_path_ds, engine_type="onnxruntime"),
            datasets=datasets_greedy_until,
            batch_size=batch_size,
            limit=2,
            gen_kwargs="max_gen_toks=16",
            use_cache=None,  # avoid saving files when running tests
        )

        out_torch = integration_eval(
            model="hf",
            model_args=f"pretrained={model_path_hf}",
            datasets=datasets_greedy_until,
            batch_size=batch_size,
            limit=2,
            gen_kwargs="max_gen_toks=16",
            use_cache=None,  # avoid saving files when running tests
        )
        self._test_same(out_onnx, out_torch, datasets_greedy_until)

    @staticmethod
    def _test_same(out_onnx, out_torch, datasets):
        datasets = datasets if isinstance(datasets, list) else [datasets]
        for dataset in datasets:
            torch_samples = out_torch.raw["samples"][dataset]
            onnx_samples = out_onnx.raw["samples"][dataset]
            for torch_sample, onnx_sample in zip(torch_samples, onnx_samples):
                print(torch_sample)
                print(onnx_sample)
                print(torch_sample["resps"], onnx_sample["resps"])
                assert torch_sample["resps"] == onnx_sample["resps"]

