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

from typing import Type

from deepsparse.v2.task import SupportedTasks, dynamic_import_task
from sparsezoo.utils.registry import (
    RegistryMixin,
    get_from_registry,
    register,
    registered_names,
)


__all__ = ["OperatorRegistry"]


class OperatorRegistry(RegistryMixin):
    """
    Register operators with given task name(s). Leverages the RegistryMixin
    functionality.
    """

    @classmethod
    def register_value(cls, operator, name):
        from deepsparse.v2.operators import Operator

        if not isinstance(name, list):
            name = [name]

        for task_name in name:
            register(Operator, operator, task_name, require_subclass=True)

        return operator

    @classmethod
    def get_task_constructor(cls, task: str) -> Type["Operator"]:  # noqa: F821
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
        from deepsparse.v2.operators import Operator

        if task.startswith("import:"):
            # dynamically import the task from a file
            task = dynamic_import_task(module_or_path=task.replace("import:", ""))
        elif task.startswith("custom"):
            # support any task that has "custom" at the beginning via the "custom" task
            task = "custom"
        else:
            task = task.lower().replace("-", "_")

        tasks = registered_names(Operator)
        # step needed to import relevant files required to load the operator
        SupportedTasks.check_register_task(task, tasks)
        return get_from_registry(Operator, task)
