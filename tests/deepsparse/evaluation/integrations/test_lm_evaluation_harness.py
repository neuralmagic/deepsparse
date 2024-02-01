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
    "datasets",
    [
        ["hellaswag"],
        # ["hellaswag", "gsm8k"],
        # "gsm8k",
        "arc_challenge",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1],  # TODO: Add test for higher batch sizes
)
@pytest.mark.skipif(
    not try_import_lm_evaluation_harness(raise_error=False),
    reason="lm_evaluation_harness not installed",
)
def test_integration_eval_onnx_matches_torch(datasets, batch_size):
    from deepsparse.evaluation.integrations.lm_evaluation_harness import (
        integration_eval,
    )

    out_torch = integration_eval(
        model="hf",
        model_args="pretrained=roneneldan/TinyStories-1M,dtype=float16",
        datasets=datasets,
        batch_size=batch_size,
        limit=2,
        use_cache=None,  # avoid saving files when running tests
    )

    out_onnx = integration_eval(
        model=create_model_from_target(
            "hf:mgoin/TinyStories-1M-ds", engine_type="onnxruntime", sequence_length=128
        ),
        datasets=datasets,
        batch_size=batch_size,
        limit=2,
        use_cache=None,  # avoid saving files when running tests
    )

    datasets = datasets if isinstance(datasets, list) else [datasets]
    for dataset in datasets:
        torch_samples = out_torch.raw["samples"][dataset]
        onnx_samples = out_onnx.raw["samples"][dataset]
        for torch_sample, onnx_sample in zip(torch_samples, onnx_samples):
            for torch_resp, onnx_resp in zip(
                torch_sample["resps"], onnx_sample["resps"]
            ):
                assert pytest.approx(torch_resp[0][0], 0.1) == onnx_resp[0][0]
                assert torch_resp[0][1] == onnx_resp[0][1]
