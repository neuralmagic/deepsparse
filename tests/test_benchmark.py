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

from typing import Dict, List

import pytest
from tests.helpers import predownload_stub, run_command


@pytest.mark.smoke
def test_benchmark_help():
    cmd = ["deepsparse.benchmark", "--help"]
    print(f"\n==== test_benchmark_help command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_benchmark_help output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "usage: deepsparse.benchmark" in res.stdout
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()


@pytest.mark.parametrize(
    ("model_stub", "additional_opts"),
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none",
            ["-shapes", "[1,128],[1,128],[1,128]"],
        ),
        (
            "zoo:bert-base_cased-squad_wikipedia_bookcorpus-pruned90",
            ["-s", "sync"],
        ),
        (
            "zoo:bert-base_cased-squad_wikipedia_bookcorpus-pruned90",
            ["-s", "async", "-nstreams", "4"],
        ),
        (
            "zoo:bert-base-wikipedia_bookcorpus-pruned90",
            ["-t", "20"],
        ),
        pytest.param(
            "zoo:bert-base-wikipedia_bookcorpus-pruned90",
            [],
            marks=pytest.mark.smoke,
        ),
        (
            "zoo:mobilenet_v1-1.0-imagenet-pruned.4block_quantized",
            ["-x", "results.json"],
        ),
        (
            "zoo:mobilenet_v1-1.0-imagenet-pruned.4block_quantized",
            ["-ncores", "4"],
        ),
        (
            "zoo:mobilenet_v1-1.0-imagenet-pruned.4block_quantized",
            ["-pin", "none"],
        ),
        (
            "zoo:yolo_v3-spp-coco-pruned",
            ["-pin", "numa", "-shapes", "[1,3,640,640]"],
        ),
        (
            "zoo:yolo_v3-spp-coco-pruned",
            ["-q"],
        ),
        (
            "zoo:yolo_v3-spp-coco-pruned",
            ["-b", "64"],
        ),
    ],
)
def test_benchmark(
    model_stub: str, additional_opts: List[str], cleanup: Dict[str, List]
):
    cmd = ["deepsparse.benchmark", model_stub, *additional_opts]
    print(f"\n==== test_benchmark command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_benchmark output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # if passing -q, check that some logging is excluded
    if "-q" in cmd:
        assert "benchmark_model.py" not in res.stdout

    # if exporting results to file, mark it for cleanup
    if "-x" in cmd:
        fn = cmd[cmd.index("-x") + 1]
        cleanup["files"].append(fn)


@pytest.mark.parametrize(
    ("model_stub"),
    [
        (
            "zoo:nlp/masked_language_modeling/bert-large/pytorch/"
            "huggingface/wikipedia_bookcorpus/pruned90-none"
        ),
    ],
)
def test_benchmark_local(model_stub: str):
    model = predownload_stub(model_stub)
    onnx_file = model.onnx_model.path
    cmd = ["deepsparse.benchmark", onnx_file]
    print(f"\n==== test_benchmark_local command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_benchmark_local output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()
