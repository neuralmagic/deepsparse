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
def register_cybertruck():
    @EvaluationRegistry.register("cybertruck_function")
    def cybertruck_function(*args, **kwargs):
        return "cybertruck"


@pytest.fixture
def register_starship():
    @EvaluationRegistry.register(alias="big_furious_rocket")
    def starship_function(*args, **kwargs):
        return "starship"


@pytest.fixture
def register_car():
    @EvaluationRegistry.register("car_types", alias=["model_y", "model_x"])
    def car_function(*args, **kwargs):
        return "car"


def test_get_cybertruck_from_registry(register_cybertruck):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.load_from_registry("cybertruck_function")
    assert eval_function() == "cybertruck"
    evaluation_registry.reset_registry()


def test_get_starship_from_registry(register_starship):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.load_from_registry("starship_function")
    eval_function_aliased = evaluation_registry.load_from_registry("big_furious_rocket")
    assert eval_function() == eval_function_aliased()
    evaluation_registry.reset_registry()


def test_get_non_existent_function(register_car, register_starship):
    evaluation_registry = EvaluationRegistry()
    with pytest.raises(KeyError):
        evaluation_registry.load_from_registry("non_existent_function")
    evaluation_registry.reset_registry()


def test_get_all_registered_names(register_car, register_starship):
    evaluation_registry = EvaluationRegistry()
    all_registered_names = evaluation_registry.registered_names()
    assert set(all_registered_names) == {
        "car_types",
        "model_x",
        "model_y",
        "starship_function",
        "big_furious_rocket",
    }
    evaluation_registry.reset_registry()


def test_get_aliased_function(register_starship):
    evaluation_registry = EvaluationRegistry()
    eval_function = evaluation_registry.load_from_registry("big_furious_rocket")
    assert eval_function() == "starship"
    evaluation_registry.reset_registry()


def test_attempt_overwrite(register_starship):
    with pytest.raises(RuntimeError):

        @EvaluationRegistry.register()
        def starship_function(*args, **kwargs):
            return "starship"
