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


def always_true(*args, **kwargs):
    return True


@EvaluationRegistry.register()
def dummy_integration(*args, **kwargs):
    return always_true


def test_evaluate():
    assert evaluate(
        target="",
        datasets="",
        integration="dummy_integration",
    )
