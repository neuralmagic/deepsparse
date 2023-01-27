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
Registry that relates the task and the data logging identifier to the
set of relevant built-in functions
"""
from collections import defaultdict
from typing import List, Union


__all__ = ["DATA_LOGGING_REGISTRY", "register"]


DATA_LOGGING_REGISTRY = defaultdict(lambda: defaultdict(str))


def register(group: Union[str, List[str]], identifier: str):
    """
    A decorator for registering the built-in function name under the
    relevant group name and data_logging_identifier.
    e.g
    ```
    @register(group="image_classification", identifier = pipeline_inputs.image)
    def some_function(image):
        ...
    ```
    this will register the function `some_function` under the
    group name `image_classification` and the data_logging_identifier
    `pipeline_inputs.image`

    :param group: The name of the group, that uses the function in question.
        Same functions may be registered under multiple groups.
    :param identifier: The name of the identifier, that the function in question
        acts on. A single identifier may be related to multiple functions.
    """

    def decorator(function):
        identifier_registry = None
        for group_name in group if isinstance(group, list) else [group]:
            task_registry = DATA_LOGGING_REGISTRY.get(group_name)
            if task_registry:
                identifier_registry = task_registry.get(identifier)
            # add the built-in function to the registry
            if identifier_registry and task_registry:
                DATA_LOGGING_REGISTRY[group_name][identifier].append(function.__name__)
            else:
                DATA_LOGGING_REGISTRY[group_name][identifier] = [function.__name__]
        return function

    return decorator
