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
from src.deepsparse.evaluation.registry import EvaluationRegistry


@pytest.fixture
def registry_with_foo():
    class Registry(EvaluationRegistry):
        pass

    @Registry.register()
    def foo(*args, **kwargs):
        return "foo"

    return Registry


@pytest.fixture
def registry_with_buzz():
    class Registry(EvaluationRegistry):
        pass

    @Registry.register(name=["buzz", "buzzer"])
    def buzz(*args, **kwargs):
        return "buzz"

    return Registry


def test_get_foo_from_registry(registry_with_foo):
    eval_function = registry_with_foo.load_from_registry("foo")
    assert eval_function() == "foo"

def test_get_multiple_buzz_from_registry(registry_with_buzz):
    eval_function_1 = registry_with_buzz.load_from_registry("buzz")
    eval_function_2 = registry_with_buzz.load_from_registry("buzzer")
    assert eval_function_1() == eval_function_2() == "buzz"

