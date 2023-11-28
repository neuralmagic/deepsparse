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
from typing import List, Union

from src.deepsparse.evaluation.integrations import (  # noqa: F401
    try_import_llm_evaluation_harness,
)
from src.deepsparse.evaluation.registry import EvaluationRegistry
from src.deepsparse.evaluation.results import Result
from src.deepsparse.pipeline import DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE


__all__ = ["evaluate"]

_LOGGER = logging.getLogger(__name__)


def evaluate(
    target: str,
    datasets: Union[str, List[str]],
    integration: str,
    engine_type: Union[
        DEEPSPARSE_ENGINE, ORT_ENGINE, TORCHSCRIPT_ENGINE, None
    ] = DEEPSPARSE_ENGINE,
    batch_size: int = 1,
    splits: Union[List[str], str, None] = None,
    metrics: Union[List[str], str, None] = None,
    **kwargs,
) -> Result:

    eval_integration = EvaluationRegistry.load_from_registry(integration)
    return eval_integration(
        target=target,
        datasets=datasets,
        engine_type=engine_type,
        batch_size=batch_size,
        splits=splits,
        metrics=metrics,
        **kwargs,
    )
