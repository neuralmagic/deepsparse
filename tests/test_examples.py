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

import os
import sys
from unittest.mock import patch

import pytest
from sparsezoo.models.classification import mobilenet_v1
from sparsezoo.objects import Model


SRC_DIRS = [
    os.path.join(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../examples"),
        dirname,
    )
    for dirname in [
        "benchmark",
        "classification",
        "detection",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import check_correctness
    import classification
    import detection
    import run_benchmark


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                mobilenet_v1,
                b,
            )
            for b in [1, 8, 64]
        ]
    ),
)
def test_check_correctness(model: Model, batch_size: int):
    m = model()
    testargs = f"""
        check_correctness.py
        {m.onnx_file.downloaded_path()}
        --batch_size {batch_size}
        """.split()

    with patch.object(sys, "argv", testargs):
        check_correctness.main()


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                mobilenet_v1,
                b,
            )
            for b in [1, 8, 64]
        ]
    ),
)
def test_run_benchmark(model: Model, batch_size: int):
    m = model()
    testargs = f"""
        run_benchmark.py
        {m.onnx_file.downloaded_path()}
        --batch_size {batch_size}
        """.split()

    with patch.object(sys, "argv", testargs):
        run_benchmark.main()


@pytest.mark.parametrize(
    "model_name, batch_size",
    (
        [
            pytest.param(
                "mobilenet_v1",
                b,
            )
            for b in [1, 8, 64]
        ]
    ),
)
def test_classification(model_name: str, batch_size: int):
    testargs = f"""
        classification.py
        {model_name}
        --batch_size {batch_size}
        """.split()

    with patch.object(sys, "argv", testargs):
        classification.main()


@pytest.mark.parametrize(
    "model_name, batch_size",
    (
        [
            pytest.param(
                "yolo_v3",
                b,
            )
            for b in [1, 8, 64]
        ]
    ),
)
def test_detection(model_name: str, batch_size: int):
    testargs = f"""
        detection.py
        {model_name}
        --batch_size {batch_size}
        """.split()

    with patch.object(sys, "argv", testargs):
        detection.main()
