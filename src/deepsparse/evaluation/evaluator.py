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
"""
The main entrypoint for evaluating a target
on a requested dataset
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE


__all__ = ["evaluate"]


@dataclass
class Metric:
    type: str
    value: float


@dataclass
class Dataset:
    type: str
    name: str
    config: str
    split: str


@dataclass
class EvalSample:
    input: Any
    output: Any


@dataclass
class Evaluation:
    # TODO: How to handle serialization of the
    # data structure (to yaml and json)
    task: str
    dataset: Dataset
    metrics: List[Metric]
    samples: List[EvalSample]


def evaluate(
    target: str,
    datasets: Union[str, List[str]],
    integration: str,
    engine_type: Union[DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None] = None,
    batch_size: int = 1,
    target_args: Optional[Dict] = None,
    engine_args: Optional[Dict] = None,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> List[Evaluation]:
    """
    :param target: The target to evaluate. Can be a path to
        a sparsezoo stub, hugging face path, or a path to a
        local directory containing a model file
    :param datasets: The datasets to evaluate on. Can be a string
        for a single dataset or a list of strings for multiple datasets.
    :param integration: The name of the evaluation integration to use.
        Must be a valid integration name that is registered in the
        evaluation registry.
    :param engine_type: The engine to use for the evaluation.
    :param batch_size: The batch size to use for the evaluation.
    :param target_args: Optional arguments to alter the
        behavior of the evaluated target.
    :param engine_args: Optional arguments to pass to the engine.
    :param splits: Specifies the name of the splits to evaluate on.
    :param metrics: Specifies the name of the metrics to evaluate on.
    :param kwargs: Additional arguments to pass to the evaluation integration.
    :return: A list of Evaluation objects containing the results of the evaluation.
    """

    # TODO: Implement a function that checks for valid target
    # TODO: Implement a function that checks for valid engine_type
    # TODO: Implement evaluation registry

    eval_integration = EvaluationRegistry.load_from_registry(integration)

    return eval_integration(
        target=target,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        target_args=target_args,
        engine_args=engine_args,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )
