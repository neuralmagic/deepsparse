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
import logging
from typing import Any, List, Optional, Union

from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Result
from deepsparse.evaluation.utils import create_model_from_target
from deepsparse.operators.engine_operator import (
    DEEPSPARSE_ENGINE,
    ORT_ENGINE,
    TORCHSCRIPT_ENGINE,
)


__all__ = ["evaluate"]

_LOGGER = logging.getLogger(__name__)


def evaluate(
    target: Any,
    datasets: Union[str, List[str]],
    integration: Optional[str] = None,
    engine_type: Union[
        DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None
    ] = DEEPSPARSE_ENGINE,
    batch_size: int = 1,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> Result:

    # if target is a string, turn it into an appropriate model/pipeline
    # otherwise assume it is a model/pipeline
    model = (
        create_model_from_target(target, engine_type)
        if isinstance(target, str)
        else target
    )

    eval_integration = EvaluationRegistry.resolve(model, datasets, integration)

    return eval_integration(
        model=model,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )
