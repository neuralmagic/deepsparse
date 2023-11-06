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


def cleanup(registry):
    registry._REGISTRY.clear()


@pytest.fixture
def register_cybertruck():
    @EvaluationRegistry.register(integration="cybertruck_function")
    def cybertruck_function(*args, **kwargs):
        return "cybertruck"


@pytest.fixture
def register_starship():
    @EvaluationRegistry.register(
        integration="starship_function", alias="big_furious_rocket"
    )
    def starship_function(*args, **kwargs):
        return "starship"


@pytest.fixture
def register_car():
    @EvaluationRegistry.register(
        integration="car_function", alias=["model_y", "model_x"]
    )
    def car_function(*args, **kwargs):
        return "car"


@pytest.fixture
def expected_print():
    return """EvaluationRegistry:
  starship_function: alias=['big_furious_rocket']
  car_function: alias=['model_y', 'model_x']"""


def test_get_cybertruck_from_registry(register_cybertruck):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.get("cybertruck_function")
    assert eval_function() == "cybertruck"
    cleanup(evaluation_registry)


def test_get_starship_from_registry(register_starship):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.get("starship_function")
    eval_function_aliased = evaluation_registry.get("big_furious_rocket")
    assert eval_function() == eval_function_aliased()
    cleanup(evaluation_registry)


def test_registry_print(register_starship, register_car, expected_print):
    evaluation_registry = EvaluationRegistry()
    registry_print = evaluation_registry.__repr__()
    assert registry_print == expected_print
    cleanup(evaluation_registry)


def test_get_non_existent_function(register_car, register_starship):
    evaluation_registry = EvaluationRegistry()
    with pytest.raises(KeyError):
        evaluation_registry.get("non_existent_function")
    cleanup(evaluation_registry)


def test_get_aliases(register_car, register_starship):
    evaluation_registry = EvaluationRegistry()
    aliases = evaluation_registry._get_available_aliases()
    assert set(aliases) == {"model_y", "model_x", "big_furious_rocket"}
    cleanup(evaluation_registry)


def test_get_aliased_function(register_starship):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.get("big_furious_rocket")
    assert eval_function() == "starship"
    cleanup(evaluation_registry)
