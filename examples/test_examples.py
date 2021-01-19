import os
import sys
from unittest.mock import patch

import pytest
from sparsezoo import Model
from sparsezoo.models.classification import mobilenet_v1

SRC_DIRS = [
    os.path.join(os.path.dirname(__file__), dirname)
    for dirname in [
        "benchmark",
        "classification",
        "detection",
    ]
]
sys.path.extend(SRC_DIRS)


if SRC_DIRS is not None:
    import classification
    import detection
    import run_benchmark
    import check_correctness


@pytest.mark.parametrize(
    "model, batch_size",
    (
        [
            pytest.param(
                mobilenet_v1,
                b,
            )
            for b in [1]  # TODO: batch override for ORT
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
        args = check_correctness.parse_args()
        result = check_correctness.main(args)