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
from pathlib import Path
from typing import List, Optional, Union

from deepsparse import Pipeline
from deepsparse.evaluation.registry import EvaluationRegistry
from deepsparse.evaluation.results import Result
from deepsparse.evaluation.utils import create_pipeline
from deepsparse.operators.engine_operator import (
    DEEPSPARSE_ENGINE,
    ORT_ENGINE,
    TORCHSCRIPT_ENGINE,
)


__all__ = ["evaluate"]

_LOGGER = logging.getLogger(__name__)


def evaluate(
    model: Union[Pipeline, Path, str],
    datasets: Union[str, List[str]],
    integration: Optional[str] = None,
    engine_type: Union[
        DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE
    ] = DEEPSPARSE_ENGINE,
    batch_size: int = 1,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> Result:

    if isinstance(model, Pipeline):
        _LOGGER.info(
            "Passed a Pipeline object into evaluate function. This will "
            "override the following arguments:"
        )
        batch_size = model.batch_size
        _LOGGER.info(f"batch_size: {batch_size}")
        engine_type = engine_type
        _LOGGER.info(f"engine_type: {engine_type}")

    # if target is a string, turn it into an appropriate pipeline
    # otherwise assume it is a pipeline
    if isinstance(model, (Path, str)):
        pipeline, kwargs = create_pipeline(model, engine_type, **kwargs)
    else:
        pipeline = model

    eval_integration = EvaluationRegistry.resolve(pipeline, datasets, integration)

    return eval_integration(
        pipeline=pipeline,
        datasets=datasets,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )
