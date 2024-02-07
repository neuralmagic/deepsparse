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

import numpy as np
from click.testing import CliRunner

import pytest
from deepsparse.evaluation.evaluator import evaluate
from deepsparse.evaluation.integrations import try_import_lm_evaluation_harness
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import (
    Dataset,
    EvalSample,
    Evaluation,
    Metric,
    Result,
)
from deepsparse.pipeline import Pipeline


@EvaluationRegistry.register()
def dummy_integration(*args, **kwargs):
    result_formatted = [
        Evaluation(
            task="task_1",
            dataset=Dataset(
                type="type_1", name="name_1", config="config_1", split="split_1"
            ),
            metrics=[Metric(name="metric_name_1", value=1.0)],
            samples=[EvalSample(input=np.array([[5]]), output=5)],
        ),
    ]
    result_raw = "dummy_result"

    return Result(formatted=result_formatted, raw=result_raw)


@pytest.fixture()
def model_path():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture()
def datasets():
    return ["hellaswag", "gsm8k"]


@pytest.fixture()
def dummy_integration_name():
    return "dummy_integration"


@pytest.fixture()
def unknown_integration_name():
    return "unknown_integration"


def test_evaluate_unknown_integration(model_path, datasets, unknown_integration_name):
    with pytest.raises(KeyError):
        evaluate(
            model=model_path,
            datasets=datasets,
            integration=unknown_integration_name,
        )


def test_evaluate(model_path, datasets, dummy_integration_name):
    result = evaluate(
        model=model_path,
        datasets=datasets,
        integration=dummy_integration_name,
    )
    assert isinstance(result, Result)


def test_evaluate_pipeline_with_kv_cache(model_path, datasets, dummy_integration_name):
    result = evaluate(
        model=Pipeline.create(model_path=model_path, task="text-generation"),
        datasets=datasets,
        integration=dummy_integration_name,
    )
    assert isinstance(result, Result)


def test_evaluate_pipeline_without_kv_cache(
    model_path, datasets, dummy_integration_name
):
    result = evaluate(
        model=Pipeline.create(
            model_path=model_path,
            task="text-generation",
            onnx_model_name="model-orig.onnx",
        ),
        datasets=datasets,
        integration=dummy_integration_name,
    )
    assert isinstance(result, Result)


@pytest.mark.skipif(
    not try_import_lm_evaluation_harness(raise_error=False),
    reason="lm_evaluation_harness not installed",
)
def test_evaluation_llm_evaluation_harness_integration_name(
    model_path,
    datasets,
):
    assert evaluate(
        model=model_path,
        # testing only on hellaswag dataset
        # to avoid long running time
        datasets=datasets[0],
        limit=1,
        integration="lm_eval_harness",
    )


@pytest.mark.parametrize("type_serialization", ["json", "yaml"])
@pytest.mark.skipif(
    tuple(map(int, sys.version.split(".")[:2])) < (3, 10),
    reason="Python version < 3.10 seems to have problems "
    "with importing functions that are decorated with "
    "click option where multiple=True",
)
def test_cli(
    tmp_path, model_path, datasets, dummy_integration_name, type_serialization
):
    from deepsparse.evaluation.cli import main

    runner = CliRunner()
    runner.invoke(
        main,
        [
            model_path,
            "--dataset",
            datasets[0],
            "--dataset",
            datasets[1],
            "--integration",
            dummy_integration_name,
            "--save_path",
            os.path.dirname(str(tmp_path)),
            "--type_serialization",
            type_serialization,
        ],
        standalone_mode=False,
    )
    # makes sure that the result file is created
    assert os.path.isfile(
        os.path.join(os.path.dirname(str(tmp_path)), f"result.{type_serialization}")
    )
