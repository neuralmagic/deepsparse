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
import pytest
from src.deepsparse.evaluation.evaluator import evaluate
from src.deepsparse.evaluation.registry import EvaluationRegistry


def return_true(*args, **kwargs):
    return True


@EvaluationRegistry.register()
def dummy_integration(*args, **kwargs):
    return return_true


def test_evaluate():
    # should fail because `dummy_integration`
    # returns a function that outputs a boolean
    # and thus is incompatible with the output type
    # of `evaluate`
    with pytest.raises(ValueError):
        evaluate(
            target="",
            datasets="",
            integration="dummy_integration",
        )


def test_evaluate_allow_original_result_structure():
    assert evaluate(
        target="",
        datasets="",
        integration="dummy_integration",
        enforce_result_structure=False,
    )
