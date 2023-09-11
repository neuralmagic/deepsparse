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

import json
import os
from typing import Dict, List

import ndjson
import pytest
from tests.helpers import predownload_stub, run_command


@pytest.mark.smoke
def test_run_inference_help():
    cmd = ["deepsparse.transformers.run_inference", "--help"]
    print(f"\n==== test_run_inference_help command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_run_inference_help output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "usage: deepsparse.transformers.run_inference" in res.stdout
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()


@pytest.mark.smoke
def test_run_inference_ner(cleanup: Dict[str, List]):
    cmd = [
        "deepsparse.transformers.run_inference",
        "--task",
        "ner",
        "--model-path",
        "zoo:bert-large-conll2003_wikipedia_bookcorpus-pruned80.4block_quantized",
        "--data",
        "tests/test_data/bert-ner-test-input.json",
        "--output-file",
        "output.json",
        "--scheduler",
        "multi",
    ]
    cleanup["files"].append("output.json")
    print(f"\n==== test_run_inference_ner command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_run_inference_ner output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # light validation of output file
    expected = "canadian"
    assert os.path.exists("output.json")
    with open("output.json") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data["predictions"][0][0]["word"] == expected


@pytest.mark.parametrize(
    ("input_format", "model_path", "local_model"),
    [
        pytest.param(
            "csv",
            "zoo:bert-base-squad_wikipedia_bookcorpus-pruned90",
            True,
            marks=pytest.mark.smoke,
        ),
        (
            "json",
            "zoo:bert-base-squad_wikipedia_bookcorpus-pruned90",
            False,
        ),
    ],
)
def test_run_inference_qa(
    input_format: str, model_path: str, local_model: bool, cleanup: Dict[str, List]
):
    if local_model:
        model = predownload_stub(model_path, copy_framework_files=True)
        model_path = model.path

    cmd = [
        "deepsparse.transformers.run_inference",
        "--task",
        "question_answering",
        "--model-path",
        model_path,
        "--data",
        f"tests/test_data/bert-qa-test-input.{input_format}",
        "--output-file",
        "output.json",
        "--scheduler",
        "single",
    ]
    cleanup["files"].append("output.json")
    print(f"\n==== test_run_inference_qa command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_run_inference_qa output ====\n{res.stdout}")

    # validate command executed successfully
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # validate output
    expected_answers = ["Snorlax", "Pikachu", "Bulbasaur"]
    assert os.path.exists("output.json")
    with open("output.json") as f:
        items = ndjson.load(f)
    for actual, expected_answer in zip(items, expected_answers):
        assert actual["answer"] == expected_answer


@pytest.mark.parametrize(
    ("input_format", "model_path", "local_model", "additional_opts"),
    [
        (
            "csv",
            "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized",
            False,
            ["--batch-size", "1", "--engine-type", "onnxruntime"],
        ),
        (
            "txt",
            "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized",
            True,
            ["--num-cores", "4", "--engine-type", "onnxruntime"],
        ),
        pytest.param(
            "csv",
            "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized",
            True,
            [],
            marks=pytest.mark.smoke,
        ),
        (
            "json",
            "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized",
            True,
            ["--batch-size", "5", "--engine-type", "deepsparse"],
        ),
        (
            "txt",
            "zoo:bert-large-mnli_wikipedia_bookcorpus-pruned80.4block_quantized",
            True,
            ["--batch-size", "10", "--num-cores", "4"],
        ),
    ],
)
def test_run_inference_sst(
    input_format: str,
    model_path: str,
    local_model: bool,
    additional_opts: List[str],
    cleanup: Dict[str, List],
):
    if local_model:
        model = predownload_stub(model_path, copy_framework_files=True)
        model_path = model.path

    cmd = [
        "deepsparse.transformers.run_inference",
        "--task",
        "text_classification",
        "--model-path",
        model_path,
        "--data",
        f"tests/test_data/bert-sst-test-input.{input_format}",
        "--output-file",
        "output.json",
        *additional_opts,
    ]
    cleanup["files"].append("output.json")
    print(f"\n==== test_run_inference_sst command ====\n{' '.join(cmd)}")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_run_inference_sst output ====\n{res.stdout}")
    assert res.returncode == 0
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()

    # light validation of output file
    # TODO: condition output validation on batch-size due to padding strategy (final
    #       input is repeated to fill in remaining batches)
    # expected = ["LABEL_1", "LABEL_0"]
    assert os.path.exists("output.json")
    # with open("output.json") as f:
    #     for idx, item in enumerate(json_lines.reader(f)):
    #         assert item[0]["label"] == expected[idx]
    # assert len(data) == 1
    # assert data[0]["label"] == expected
