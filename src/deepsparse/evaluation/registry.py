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

from typing import Any, Callable, List, Optional, Union

from sparsezoo.utils.registry import RegistryMixin


__all__ = ["EvaluationRegistry"]


class EvaluationRegistry(RegistryMixin):
    """
    Extends the RegistryMixin to enable registering
    and loading of evaluation functions.
    """

    @classmethod
    def load_from_registry(cls, name: str) -> Callable[..., Any]:
        return cls.get_value_from_registry(name=name)

    @classmethod
    def resolve(
        cls, model: Any,
             datasets: Union[str, List[str]],
             integration: Optional[str]
    ) -> Callable[..., Any]:
        """
        Chooses an evaluation function from the registry based on the target,
        datasets and integration.

        If integration is specified, attempts to load the evaluation function
        from the registry.

        Currently:
            if integration is not specified, default to 'llm-evaluation-harness'
            for llm-type models. If the model is not a llm-type model, raise an error.
        """
        if integration is not None:
            try:
                integration_eval = cls.load_from_registry(name=integration)
                return integration_eval
            except KeyError:
                pass
        # resolve
        return cls.get_value_from_registry(name="llm-evaluation-harness")
