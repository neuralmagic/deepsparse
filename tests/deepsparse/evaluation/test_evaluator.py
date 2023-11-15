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

import numpy as np
from click.testing import CliRunner

import pytest
from src.deepsparse.evaluation.evaluator import main
from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import Dataset, EvalSample, Evaluation, Metric


@EvaluationRegistry.register()
def return_zero_integration(*args, **kwargs):
    return True


@EvaluationRegistry.register()
def dummy_integration(*args, **kwargs):
    return [
        Evaluation(
            task="task_1",
            dataset=Dataset(
                type="type_1", name="name_1", config="config_1", split="split_1"
            ),
            metrics=[Metric(name="metric_name_1", value=1.0)],
            samples=[EvalSample(input=np.array([[5]]), output=5)],
        ),
    ]


@pytest.fixture()
def target():
    return "hf:mgoin/TinyStories-1M-deepsparse"


@pytest.fixture()
def datasets():
    return "hellaswag"


@pytest.fixture()
def llm_evaluation_harness_integration_name():
    return "llm_evaluation_harness"


@pytest.fixture()
def dummy_integration_name():
    return "dummy_integration"


@pytest.fixture()
def return_zero_integration_name():
    return "return_zero_integration"


@pytest.fixture()
def unknown_integration_name():
    return "unknown_integration"


def test_evaluation_unknown_integration(target, datasets, unknown_integration_name):
    # When attempting to load an unknown integration, should raise a KeyError
    # (unable to find the integration in the registry)
    runner = CliRunner()
    out = runner.invoke(
        main,
        [
            "--target",
            target,
            "--datasets",
            datasets,
            "--integration",
            unknown_integration_name,
        ],
    )
    assert isinstance(out.exception, KeyError)


def test_evaluation_dummy_integration_wrong_structure(
    target, datasets, return_zero_integration_name
):
    # Dummy integration returns a boolean, but the evaluation script
    # expects a List[Evaluation] and thus should raise a ValueError
    runner = CliRunner()
    out = runner.invoke(
        main,
        [
            "--target",
            target,
            "--datasets",
            datasets,
            "--integration",
        ],
    )
    assert isinstance(out.exception, ValueError)


def test_evaluation_dummy_integration_arbitrary_structure(
    target, datasets, return_zero_integration_name
):
    # Dummy integration returns a boolean and the evaluation script
    # now allows for that, so should not raise a ValueError
    runner = CliRunner()
    out = runner.invoke(
        main,
        [
            "--target",
            target,
            "--datasets",
            datasets,
            "--integration",
            return_zero_integration_name,
            "--enforce-result-structure",
            False,
        ],
        standalone_mode=False,
    )
    assert isinstance(out.return_value, True)


@pytest.mark.parametrize("type_serialization", ["json", "yaml"])
def test_evaluation_serialize_result(
    tmp_path, target, datasets, dummy_integration_name, type_serialization
):
    # Dummy integration returns a boolean and the evaluation script
    # now allows for that, so should not raise a ValueError
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
    assert len(out.output) == 532
    assert os.path.isfile(
        os.path.join(os.path.dirname(str(tmp_path)), f"result.{type_serialization}")
    )


def test_evaluation_llm_evaluation_harness_integration_name(
    target, datasets, llm_evaluation_harness_integration_name
):
    runner = CliRunner()
    out = runner.invoke(
        main,
        [
            "--target",
            target,
            "--datasets",
            datasets,
            "--integration",
            llm_evaluation_harness_integration_name,
        ],
    )
    print(out)
