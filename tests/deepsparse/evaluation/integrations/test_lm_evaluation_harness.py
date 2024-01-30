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
    "pipeline, model_torch",
    [
        (
            create_pipeline(
                "hf:mgoin/TinyStories-1M-deepsparse", engine_type="onnxruntime"
            ),
            create_pipeline("roneneldan/TinyStories-1M"),
        )
    ],
)
@pytest.mark.parametrize(
    "datasets",
    [
        ["hellaswag"],
        ["hellaswag", "gsm8k"],
        "gsm8k",
        "arc_challenge",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 3],
)
class TestLMEvaluationHarness:
    @pytest.mark.skipif(
        not try_import_lm_evaluation_harness(raise_error=False),
        reason="lm_evaluation_harness not installed",
    )
    def test_integration_eval_onnx_matches_torch(
        self, pipeline, model_torch, datasets, batch_size
    ):
        from deepsparse.evaluation.integrations.lm_evaluation_harness import (
            integration_eval,
        )

        out_torch = integration_eval(
            model=model_torch,
            datasets=datasets,
            batch_size=batch_size,
            limit=5,
            no_cache=True,  # avoid saving files when running tests
        )
        out_onnx = integration_eval(
            model=pipeline,
            datasets=datasets,
            batch_size=batch_size,
            limit=5,
            no_cache=True,  # avoid saving files when running tests
        )
        out_onnx = out_onnx.raw["output"]
        out_torch = out_torch.raw["output"]

        assert out_onnx["results"] == out_torch["results"]
