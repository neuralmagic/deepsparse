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

from collections import defaultdict


_FUNCTIONS_REGISTRY = defaultdict(lambda: defaultdict(str))


def register(task: str, identifier: str):
    """
    A decorator for registering the function name under the task and identifier strings.
    e.g
    ```
    _FUNCTIONS_REGISTRY = {"task1": {"identifier1": [func_name_1, func_name_2]},
                           "task2": {"identifier2": [func_name_1, func_name_3]},
                                    {"identifier1": [func_name_1]}}

    :param task: The name of the task, that uses the function in question.
        Same function names may be registered under multiple
        tasks or multiple identifiers.
    :param identifier: The name of the identifier, that the function in question
        acts on. A single identifier may be related to multiple functions.
    :return:
    """

    def decorator(f):
        task_registry = _FUNCTIONS_REGISTRY.get(task)
        if task_registry is None:
            # if the task is not in the registry, add it
            _FUNCTIONS_REGISTRY[task][identifier] = [f.__name__]
        else:
            _FUNCTIONS_REGISTRY[task][identifier].append(f.__name__)

        return f

    return decorator
