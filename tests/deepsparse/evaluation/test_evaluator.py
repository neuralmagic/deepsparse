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
import shutil

import numpy as np
from click.testing import CliRunner

import pytest
from src.deepsparse.evaluation.evaluator import evaluate, main
from src.deepsparse.evaluation.integrations import try_import_llm_evaluation_harness
from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import (
    Dataset,
    EvalSample,
    Evaluation,
    Metric,
    Result,
)


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
def target():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture()
def datasets():
    return ["hellaswag", "gsm8k"]


@pytest.fixture()
def llm_evaluation_harness_integration_name():
    return "llm_evaluation_harness"


@pytest.fixture()
def dummy_integration_name():
    return "dummy_integration"


@pytest.fixture()
def unknown_integration_name():
    return "unknown_integration"


def test_evaluate_unknown_integration(target, datasets, unknown_integration_name):
    with pytest.raises(KeyError):
        evaluate(
            target=target,
            datasets=datasets,
            integration=unknown_integration_name,
        )


def test_evaluate(target, datasets, dummy_integration_name):
    result = evaluate(
        target=target,
        datasets=datasets,
        integration=dummy_integration_name,
    )
    assert isinstance(result, Result)


@pytest.mark.parametrize("type_serialization", ["json", "yaml"])
def test_cli(tmp_path, target, datasets, dummy_integration_name, type_serialization):
    runner = CliRunner()
    out = runner.invoke(
        main,
        [
            "--target",
            target,
            "--datasets",
            datasets,
            "--integration",
            dummy_integration_name,
            "--save_path",
            os.path.dirname(str(tmp_path)),
            "--type_serialization",
            type_serialization,
        ],
        standalone_mode=False,
    )
    assert isinstance(out.output, str)
    # 532 is the length of the json/yaml string
    assert len(out.output) == 532
    # makes sure that the result file is created
    assert os.path.isfile(
        os.path.join(os.path.dirname(str(tmp_path)), f"result.{type_serialization}")
    )


@pytest.fixture(scope="function")
def delete_created_files():
    yield
    try:
        shutil.rmtree(os.path.join(os.getcwd(), "tests/testdata"))
    except Exception:
        pass


@pytest.mark.skipif(
    not try_import_llm_evaluation_harness(raise_error=False),
    reason="llm_evaluation_harness not installed",
)
def test_evaluation_llm_evaluation_harness_integration_name(
    target, datasets, llm_evaluation_harness_integration_name, delete_created_files
):
    # import to trigger the registration of the
    # evaluation function for `llm_evaluation_harness`
    from src.deepsparse.evaluation.integrations.llm_evaluation_harness import (  # noqa F401 E501
        integration_eval,
    )

    assert evaluate(
        target=target,
        datasets=datasets,
        limit=2,
        no_cache=True,
        integration="llm_evaluation_harness",
    )
