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
from src.deepsparse.evaluation.integrations.llm_evaluation_harness import (
    integration_eval,
)


try:
    import lm_eval  # noqa F401

    lm_eval_not_installed = False
except Exception:
    lm_eval_not_installed = True


@pytest.mark.parametrize(
    "target_onnx, target_torch",
    [
        (
            "hf:mgoin/TinyStories-1M-deepsparse",
            "roneneldan/TinyStories-1M",
        )
    ],
)
# TODO: Write tests for other datasets relevant for us
# TODO: gsm8k cannot be tested, because it explicitly requires
# a model that follows specific output format
@pytest.mark.parametrize(
    "datasets",
    [
        ["hellaswag"],
        ["hellaswag", "gsm8k"],
        "gsm8k",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 3],
)
@pytest.mark.skipif(lm_eval_not_installed, reason="Requires lm_eval installed")
def test_integration_eval_onnx_matches_torch(
    target_onnx, target_torch, datasets, batch_size
):
    out_torch = integration_eval(
        target=target_torch,
        datasets=datasets,
        batch_size=batch_size,
        target_args={},
        limit=10,
        engine_type="torch",
        no_cache=True,  # avoid saving files when running tests
    )
    out_onnx = integration_eval(
        target=target_onnx,
        datasets=datasets,
        batch_size=batch_size,
        target_args={},
        limit=10,
        engine_type="onnxruntime",
        no_cache=True,  # avoid saving files when running tests
    )
    datasets = datasets if isinstance(datasets, list) else [datasets]
    for dataset in datasets:
        for key, value in out_torch["results"][dataset].items():
            assert value == out_onnx["results"][dataset][key]
