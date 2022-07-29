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

from subprocess import PIPE, STDOUT, Popen
from typing import Dict, List

import requests

import pytest

from .helpers import predownload_stub, run_command, wait_for_server


# TODO: Update to either use sparsezoo stubs or pre-download models on-the-fly to test
#       with local model files
# TODO: Include additional opts/etc. in tests


@pytest.mark.smoke
def test_server_help():
    cmd = ["deepsparse.server", "--help"]
    print(f"\n==== test_server_help command ====\n{' '.join(cmd)}\n==== ====")
    res = run_command(cmd)
    if res.stdout is not None:
        print(f"\n==== test_server_help output ====\n{res.stdout}\n==== ====")
    assert res.returncode == 0
    assert "Usage:" in res.stdout
    assert "error" not in res.stdout.lower()
    assert "fail" not in res.stdout.lower()


def test_server_ner(cleanup: Dict[str, List]):
    cmd = [
        "deepsparse.server",
        "--task",
        "ner",
        "--model_path",
        "zoo:nlp/token_classification/bert-base/pytorch/huggingface/conll2003/"
        "12layer_pruned80_quant-none-vnni",
    ]
    print(f"\n==== test_server_ner command ====\n{' '.join(cmd)}\n==== ====")
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    cleanup["processes"].append(proc)
    is_up = wait_for_server("http://localhost:5543/docs", 60)

    if not is_up:
        proc.terminate()
        print(
            "\n==== test_server_ner output ====\n"
            f"{proc.stdout.read().decode('utf-8')}\n==== ===="
        )
        assert is_up is True, "failed to start server"

    # run a sample request to the endpoint and validate the response
    try:
        payload = {"inputs": "Red color"}
        rsp = requests.post("http://localhost:5543/predict", json=payload)
        rsp.raise_for_status()
        assert rsp.status_code == 200
        assert rsp.headers["content-type"] == "application/json"
        assert len(rsp.json()) > 0
        predictions = rsp.json()["predictions"]
        assert "word" in predictions[0][0]
        assert predictions[0][0]["word"] == "red"
    except:  # noqa
        # end server proc if we encounter failures during request/response validation
        proc.terminate()
        raise

    proc.terminate()
    returncode = proc.wait()
    output = proc.stdout.read().decode("utf-8")
    print(f"\n==== test_server_ner output ====\n{output}\n==== ====")
    assert returncode == 0
    assert "error" not in output.lower()
    assert "fail" not in output.lower()


@pytest.mark.smoke
def test_server_qa(cleanup: Dict[str, List]):
    cmd = [
        "deepsparse.server",
        "--task",
        "question_answering",
        "--model_path",
        "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/"
        "pruned_6layers-aggressive_98",
    ]
    print(f"\n==== test_server_qa command ====\n{' '.join(cmd)}\n==== ====")
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    cleanup["processes"].append(proc)
    is_up = wait_for_server("http://localhost:5543/docs", 60)

    if not is_up:
        proc.terminate()
        print(
            "\n==== test_server_qa output ====\n"
            f"{proc.stdout.read().decode('utf-8')}\n==== ===="
        )
        assert is_up is True, "failed to start server"

    # run a sample request to the endpoint and validate the response
    try:
        payload = {"question": "What's my name", "context": "My name is Snorlax"}
        rsp = requests.post("http://localhost:5543/predict", json=payload)
        rsp.raise_for_status()
        assert rsp.status_code == 200
        assert rsp.headers["content-type"] == "application/json"
        assert "answer" in rsp.json()
        assert rsp.json()["answer"] == "Snorlax"
    except:  # noqa
        # end server proc if we encounter failures during request/response validation
        proc.terminate()
        raise

    proc.terminate()
    returncode = proc.wait()
    output = proc.stdout.read().decode("utf-8")
    print(f"\n==== test_server_qa output ====\n{output}\n==== ====")
    assert returncode == 0
    assert "error" not in output.lower()
    assert "fail" not in output.lower()


def test_server_qa_config_file(cleanup: Dict[str, List]):
    cmd = [
        "deepsparse.server",
        "--config_file",
        "tests/test_data/deepsparse-server-config.yaml",
    ]
    print(f"\n==== test_server_qa_config_file command ====\n{' '.join(cmd)}\n==== ====")
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    cleanup["processes"].append(proc)
    # longer timeout due to potentially downloading two separate models, one dense
    is_up = wait_for_server("http://localhost:5543/docs", 180)

    if not is_up:
        proc.terminate()
        print(
            "\n==== test_server_qa_config_file output ====\n"
            f"{proc.stdout.read().decode('utf-8')}\n==== ===="
        )
        assert is_up is True, "failed to start server"

    # run sample requests to the endpoint for both models and validate the responses
    try:
        payload = {"question": "What's my name", "context": "My name is Snorlax"}
        rsp = requests.post(
            "http://localhost:5543/predict/question_answering/dense",
            json=payload,
        )
        rsp.raise_for_status()
        assert rsp.status_code == 200
        assert rsp.headers["content-type"] == "application/json"
        assert "answer" in rsp.json()
        assert rsp.json()["answer"] == "Snorlax"
    except:  # noqa
        # end server proc if we encounter failures during request/response validation
        proc.terminate()
        raise

    try:
        payload = {"question": "What's my name", "context": "My name is Snorlax"}
        rsp = requests.post(
            "http://localhost:5543/predict/question_answering/sparse_quantized",
            json=payload,
        )
        rsp.raise_for_status()
        assert rsp.status_code == 200
        assert rsp.headers["content-type"] == "application/json"
        assert "answer" in rsp.json()
        assert rsp.json()["answer"] == "Snorlax"
    except:  # noqa
        # end server proc if we encounter failures during request/response validation
        proc.terminate()
        raise

    proc.terminate()
    returncode = proc.wait()
    output = proc.stdout.read().decode("utf-8")
    print(f"\n==== test_server_qa_config_file output ====\n{output}\n==== ====")
    assert returncode == 0
    assert "error" not in output.lower()
    assert "fail" not in output.lower()


def test_server_sst(cleanup: Dict[str, List]):
    stub = "zoo:nlp/text_classification/bert-base/pytorch/huggingface/sst2/base-none"
    model = predownload_stub(stub, copy_framework_files=True)
    cmd = [
        "deepsparse.server",
        "--task",
        "sentiment_analysis",
        "--model_path",
        model.onnx_file.dir_path,
    ]
    print(f"\n==== test_server_sst command ====\n{' '.join(cmd)}\n==== ====")
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT)
    cleanup["processes"].append(proc)
    is_up = wait_for_server("http://localhost:5543/docs", 60)

    if not is_up:
        proc.terminate()
        print(
            "\n==== test_server_sst output ====\n"
            f"{proc.stdout.read().decode('utf-8')}\n==== ===="
        )
        assert is_up is True, "failed to start server"

    # run a sample request to the endpoint and validate the response
    try:
        payload = {"sequences": "this is good"}
        rsp = requests.post("http://localhost:5543/predict", json=payload)
        rsp.raise_for_status()
        assert rsp.status_code == 200
        assert rsp.headers["content-type"] == "application/json"
        assert len(rsp.json()) > 0
        rsp_body = rsp.json()
        assert "labels" in rsp_body
        assert rsp_body["labels"][0] == "LABEL_1"
    except:  # noqa
        # end server proc if we encounter failures during request/response validation
        proc.terminate()
        output = proc.stdout.read().decode("utf-8")
        print(f"\n==== test_server_sst output ====\n{output}\n==== ====")
        raise

    proc.terminate()
    returncode = proc.wait()
    output = proc.stdout.read().decode("utf-8")
    print(f"\n==== test_server_sst output ====\n{output}\n==== ====")
    assert returncode == 0
    assert "error" not in output.lower()
    assert "fail" not in output.lower()
