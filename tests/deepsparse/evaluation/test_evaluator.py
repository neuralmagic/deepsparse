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

from src.deepsparse.evaluation.evaluator import evaluate
from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import Dataset, EvalSample, Evaluation, Metric


def return_true(*args, **kwargs):
    return True


def return_list_of_evaluations(*args, **kwargs):
    return [
        Evaluation(
            task="task_1",
            dataset=Dataset(
                type="type_1", name="name_1", config="config_1", split="split_1"
            ),
            metrics=[Metric(name="metric_name_1", value=1.0)],
            samples=[EvalSample(input=5, output=5)],
        )
    ]


@EvaluationRegistry.register()
def dummy_integration_returns_boolean(*args, **kwargs):
    return return_true


@EvaluationRegistry.register()
def dummy_integration(*args, **kwargs):
    return return_list_of_evaluations


def test_evaluate():
    result = evaluate(
        target="",
        datasets="",
        integration="dummy_integration",
    )
    assert [isinstance(result_, Evaluation) for result_ in result]


def test_evaluate_preserve_original_result_structure():
    result = evaluate(
        target="",
        datasets="",
        integration="dummy_integration_returns_boolean",
    )
    assert isinstance(result, bool)
