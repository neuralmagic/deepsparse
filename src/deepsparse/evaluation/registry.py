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


def _merge_name_alias(
    name: Optional[str], alias: Union[str, List[str], None]
) -> List[str]:
    """
    Merge the name and alias into a single list of names
    """
    name = [name]
    if alias:
        if not isinstance(alias, list):
            alias = [alias]
        name.extend(alias)
    return name


class EvaluationRegistry(RegistryMixin):
    """
    Extends the RegistryMixin to enable registering and loading of evaluation
    functions. Adds the ability to register a function under multiple names.
    Since functions are stateless, we can register the same function under
    multiple names and load it from the registry using any of the names.

    Example:
    To register function `get_rodent` as under names `get_rodent` and `get_squirrel`:

    ```python
    @EvaluationRegistry.register(alias="get_squirrel")
    def get_rodent(*args, **kwargs):
        return "squirrel"
    ```

    Alternatively (explicitly specifying the name):

    ```python
    @EvaluationRegistry.register(name="get_rodent", alias="get_squirrel")
    def this_function_name_will_not_be_registered(*args, **kwargs):
        return "squirrel"
    ```
    """

    @classmethod
    def register(
        cls, name: Optional[str] = None, alias: Union[str, List[str], None] = None
    ):
        def decorator(value: Any):
            cls.register_value(value, names=_merge_name_alias(name, alias))
            return value

        return decorator

    @classmethod
    def register_value(cls, value: Any, names: List[str]):
        for name in names:
            super().register_value(value, name)

    @classmethod
    def load_from_registry(cls, name: str) -> Callable[..., Any]:
        return cls.get_value_from_registry(name=name)
