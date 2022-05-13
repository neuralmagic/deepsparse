from typing import Dict, List

import pytest

from helpers import predownload_stub, run_command


@pytest.mark.cli
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


@pytest.mark.cli
@pytest.mark.parametrize(
    ("model_stub", "additional_opts"),
    [
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/base-none",
            ["-shapes", "[1,128],[1,128],[1,128]"],
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned-aggressive_98",
            ["-s", "sync"],
        ),
        (
            "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/pruned_quant-aggressive_95",
            ["-s", "async", "-nstreams", "10"],
        ),
        (
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/base-none",
            ["-t", "20"],
        ),
        (
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned90-none",
            [],
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/base-none",
            ["-x", "results.json"],
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned-moderate",
            ["-ncores", "4"],
        ),
        (
            "zoo:cv/classification/resnet_v1-50/pytorch/sparseml/imagenet/pruned_quant-moderate",
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
            "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned_quant-aggressive_94",
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


@pytest.mark.cli
@pytest.mark.parametrize(
    ("model_stub"),
    [
        (
            "zoo:nlp/masked_language_modeling/bert-base/pytorch/huggingface/bookcorpus_wikitext/12layer_pruned90-none"
        ),
    ],
)
def test_benchmark_local(model_stub: str):
    model = predownload_stub(model_stub)
    onnx_file = model.onnx_file.path
    cmd = ["deepsparse.benchmark", onnx_file]
    print(f"\n==== test_benchmark_local command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_benchmark_local output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()
