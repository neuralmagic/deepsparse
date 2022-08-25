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
from src.deepsparse.benchmark.benchmark_model import parse_export_dict_engine_key
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
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "pruned-aggressive_98",
            ["-s", "sync"],
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
            "pruned_quant-aggressive_95",
            ["-s", "async", "-nstreams", "4"],
        ),
        (
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
            "bookcorpus_wikitext/base-none",
            ["-t", "20"],
        ),
        pytest.param(
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
            "bookcorpus_wikitext/12layer_pruned90-none",
            [],
            marks=pytest.mark.smoke,
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
            ["-x", "results.json"],
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned-moderate",
            ["-ncores", "4"],
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/"
            "pruned_quant-moderate",
            ["-pin", "none"],
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none",
            ["-pin", "numa"],
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned-aggressive_96",
            ["-q"],
        ),
        (
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/"
            "pruned_quant-aggressive_94",
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
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/"
            "bookcorpus_wikitext/12layer_pruned90-none"
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


@pytest.mark.parametrize(
    ("engine_key", "expected"),
    [
        (
            "deepsparse.engine.Engine:\n\tversion: 1.1.0.20220805\n\tcommit: 8f2ea328"
            "\n\tonnx_file_path: /home/george/.cache/sparsezoo/61686faf-c724-42f9"
            "-b9ae-749266e3f669/model.onnx\n\tbatch_size: 1\n\tnum_cores: 18\n\t"
            "num_streams: 9\n\tscheduler: Scheduler.multi_stream\n\t"
            "cpu_avx_type: avx512"
            "\n\tcpu_vnni: False",
            {
                "engine": "deepsparse.engine.Engine",
                "version": "1.1.0.20220805",
                "commit": "8f2ea328",
                "onnx_file_path": (
                    "/home/george/.cache/sparsezoo/"
                    "61686faf-c724-42f9-b9ae-749266e3f669/model.onnx"
                ),
                "batch_size": 1,
                "num_cores": 18,
                "num_streams": 9,
                "scheduler": "Scheduler.multi_stream",
                "cpu_avx_type": "avx512",
                "cpu_vnni": False,
                "persistent_key": "this entry should be persistent",
            },
        ),
    ],
)
def test_parse_export_dict_engine_key(engine_key: str, expected: Dict):
    input_payload = {
        "engine": engine_key,
        "persistent_key": "this entry should be persistent",
    }
    response = parse_export_dict_engine_key(input_payload)
    assert (
        expected == response
    ), f"Error in parsing str(model). Expected {expected}\ngot {response}"
