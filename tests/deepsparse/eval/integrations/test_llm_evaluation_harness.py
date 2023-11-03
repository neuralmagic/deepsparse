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
from src.deepsparse.eval.integrations.llm_evaluation_harness import integration_eval


@pytest.mark.parametrize(
    "target_onnx, target_torch,datasets, batch_size",
    [
        (
            "hf:mgoin/TinyStories-1M-deepsparse",
            "roneneldan/TinyStories-1M",
            ["hellaswag"],
            3,
        )
    ],
)
def test_integration_eval_onnx_matches_torch(
    target_onnx, target_torch, datasets, batch_size
):
    out_torch = integration_eval(
        target=target_torch,
        datasets=datasets,
        batch_size=batch_size,
        target_args={},
        limit=5,
        engine_type="torch",
        splits=None,
        metrics=None,
        engine_args=None,
        no_cache=True,  # avoid saving files when running tests
    )
    out_onnx = integration_eval(
        target=target_onnx,
        datasets=datasets,
        batch_size=batch_size,
        target_args={},
        limit=5,
        engine_type="onnxruntime",
        splits=None,
        metrics=None,
        engine_args=None,
        no_cache=True,  # avoid saving files when running tests
    )
    for dataset in datasets:
        for key, value in out_torch["results"][dataset].items():
            assert value == out_onnx["results"][dataset][key]
