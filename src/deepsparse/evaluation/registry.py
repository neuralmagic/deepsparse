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
Implementation of a registry for evaluation functions
"""
import logging
from typing import Callable, List, Optional, Union

from deepsparse import Pipeline
from sparsezoo.utils.registry import RegistryMixin


__all__ = ["EvaluationRegistry"]

_LOGGER = logging.getLogger(__name__)


class EvaluationRegistry(RegistryMixin):
    """
    Extends the RegistryMixin to enable registering
    and loading of evaluation functions.
    """

    @classmethod
    def load_from_registry(cls, name: str) -> Callable[..., "Result"]:  # noqa: F821
        return cls.get_value_from_registry(name=name)

    @classmethod
    def resolve(
        cls,
        pipeline: Pipeline,
        datasets: Union[str, List[str]],
        integration: Optional[str] = None,
    ) -> Callable[..., "Result"]:  # noqa: F821
        """
        Chooses an evaluation function from the registry based on the target,
        datasets and integration.

        If integration is specified, attempts to load the evaluation function
        from the registry.
        """
        from deepsparse.evaluation.utils import (
            potentially_check_dependency_import,
            resolve_integration,
        )

        if integration is None:
            _LOGGER.info(
                "No integration specified, inferring the evaluation "
                "function from the input arguments..."
            )
            integration = resolve_integration(pipeline, datasets)

            if integration is None:
                raise ValueError(
                    "Unable to resolve an evaluation function for the given model. "
                    "Specify an integration name or use a pipeline that is supported "
                )
            _LOGGER.info(f"Inferred the evaluation function: {integration}")

        potentially_check_dependency_import(integration)

        try:
            return cls.load_from_registry(name=integration)
        except KeyError as err:
            raise KeyError(err)
