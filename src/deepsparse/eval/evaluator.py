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
The main entrypoint for evaluating a model
or a Pipeline on a requested dataset
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, Pipeline
from src.deepsparse.eval.registry import EVAL_REGISTRY


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
    task: str
    dataset: Dataset
    metrics: List[Metric]
    samples: List[EvalSample]


def eval(
    target: Union["Module", Pipeline],
    datasets: Union[str, List[str]],
    integration: str,
    batch_size: int = 1,
    target_args: Optional[Dict] = None,
    engine: Union[None, DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE] = None,
    engine_args: Optional[Dict] = None,
    splits=None,
    metrics=None,
    **kwargs,
) -> List[Evaluation]:
    # TODO: Decide on the final types of arguments later

    # TODO: Implement registry
    eval_integration = EVAL_REGISTRY.resolve(target, integration, datasets)

    return eval_integration(
        target=target,
        target_args=target_args,
        datasets=datasets,
        splits=splits,
        metrics=metrics,
        batch_size=batch_size,
        engine=engine,
        engine_args=engine_args,
        **kwargs,
    )
