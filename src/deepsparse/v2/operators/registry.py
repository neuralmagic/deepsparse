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

from typing import List, Optional, Type

from deepsparse.v2.task import SupportedTasks, dynamic_import_task


_REGISTERED_OPERATORS = {}

__all__ = ["OperatorRegistry"]


class OperatorRegistry:
    def register(task: str, task_aliases: Optional[List[str]] = None):
        from deepsparse.v2.operators import Operator

        """
        Decorator to register an operator with its task name and its aliases. The
        registered names can be used to load the operator through `Operator.create()`

        Multiple operators may not have the same task name. An error will
        be raised if two different operators attempt to register the same task name

        :param task: main task name of this operator
        :param task_aliases: list of extra task names that may be used to reference
            this operator. Default is None
        """
        task_names = [task]
        if task_aliases:
            task_names.extend(task_aliases)

        task_names = [task_name.lower().replace("-", "_") for task_name in task_names]

        def _register_task(task_name, operator):
            if task_name in _REGISTERED_OPERATORS and (
                operator is not _REGISTERED_OPERATORS[task_name]
            ):
                raise RuntimeError(
                    f"task {task_name} already registered by OperatorRegistry.register "
                    f"attempting to register operator: {operator}, but"
                    f"operator: {_REGISTERED_OPERATORS[task_name]}, already registered"
                )
            _REGISTERED_OPERATORS[task_name] = operator

        def _register_operator(operator: Operator):
            if not issubclass(operator, Operator):
                raise RuntimeError(
                    f"Attempting to register operator {operator}. "
                    f"Registered operators must inherit from {Operator}"
                )
            for task_name in task_names:
                _register_task(task_name, operator)

            # set task and task_aliases as class level property
            operator.task = task
            operator.task_aliases = task_aliases

            return operator

        return _register_operator

    @staticmethod
    def get_task_constructor(task: str) -> Type["Operator"]:
        """
        This function retrieves the class previously registered via
        `OperatorRegistry.register` for `task`.

        If `task` starts with "import:", it is treated as a module to be imported,
        and retrieves the task via the `TASK` attribute of the imported module.

        If `task` starts with "custom", then it is mapped to the "custom" task.

        :param task: The task name to get the constructor for
        :return: The class registered to `task`
        :raises ValueError: if `task` was not registered via `OperatorRegistry.register`
        """
        if task.startswith("import:"):
            # dynamically import the task from a file
            task = dynamic_import_task(module_or_path=task.replace("import:", ""))
        elif task.startswith("custom"):
            # support any task that has "custom" at the beginning via the "custom" task
            task = "custom"
        else:
            task = task.lower().replace("-", "_")

        # step needed to import relevant files required to load the operator
        SupportedTasks.check_register_task(task, _REGISTERED_OPERATORS.keys())

        if task not in _REGISTERED_OPERATORS:
            raise ValueError(
                f"Unknown Operator task {task}. Operators tasks should be "
                "must be declared with the OperatorRegistry.register decorator. "
                f"Currently registered operators: {list(_REGISTERED_OPERATORS.keys())}"
            )

        return _REGISTERED_OPERATORS[task]
