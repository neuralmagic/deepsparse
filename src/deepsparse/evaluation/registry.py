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

import textwrap
from collections import defaultdict
from typing import Any, Callable, List, Optional, Union

from pydantic import BaseModel, Field




class EvaluationRegistryEntry(BaseModel):
    alias: Union[List[str], None] = Field(
        default=None,
        description="Alternative name for the evaluation function integration.",
    )
    function: Callable[..., Any] = Field(
        description="The evaluation function particular to the integration",
        repr=False,
    )
    # TODO: How do we envision using it?
    error_msg: Optional[str] = Field(
        default=None,
        description="The additional error message to "
        "display if the evaluation function fails",
        repr=False,
    )


class EvaluationRegistry:
    _REGISTRY = defaultdict(lambda: EvaluationRegistryEntry)

    @classmethod
    def register(
        cls,
        integration: str,
        alias: Union[str, List[str], None] = None,
        error_msg: Optional[str] = None,
    ) -> Callable[..., Any]:
        """
        # TODO: Add docstrings
        """
        if alias is not None:
            alias = alias if isinstance(alias, list) else [alias]

        def decorator(function):
            if cls._REGISTRY.get(integration):
                raise ValueError(
                    f"Integration name: {integration} is already "
                    f"registered for a function: {function.__name__}"
                )
            cls._REGISTRY[integration] = EvaluationRegistryEntry(
                alias=alias, function=function, error_msg=error_msg
            )
            return function

        return decorator

    def get(self, integration: str) -> Callable[..., Any]:
        # TODO: Reconcile with the requirement
        # "EVAL_REGISTRY.resolve(target, integration, datasets)"

        # get the evaluation function by integration name
        entry_found = self._REGISTRY.get(integration)
        if entry_found:
            return entry_found.function

        # otherwise get the evaluation function by integration name
        entry_found = self._check_for_alias(integration)
        if entry_found is None:
            raise KeyError(
                f"Integration {integration} not found in the registry. "
                f"Available integrations: {list(self._REGISTRY.keys())}. "
                f"Available aliases: {self._get_available_aliases()}"
            )
        return entry_found.function

    def __repr__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self._print_registry_entries(self._REGISTRY)}"
        )

    def _get_available_aliases(self) -> List[str]:
        all_aliases = []
        for entry in self._REGISTRY.values():
            if entry.alias is None:
                continue
            all_aliases.extend(entry.alias)
        return all_aliases

    def _check_for_alias(self, integration: str) -> Optional[EvaluationRegistryEntry]:
        return next(
            (
                entry
                for entry in self._REGISTRY.values()
                if entry.alias is not None and integration in entry.alias
            ),
            None,
        )

    @staticmethod
    def _print_registry_entries(registry: dict) -> str:
        return textwrap.indent(
            "\n".join([f"{key}: {value}" for key, value in registry.items()]),
            prefix="  ",
        )
