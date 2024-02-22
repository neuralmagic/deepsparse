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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel

from deepsparse.operators.registry import OperatorRegistry
from deepsparse.utils import InferenceState


__all__ = ["Operator"]


class Operator(ABC):
    """
    Base operator class - an operator should be defined for each atomic, functional
    part of the pipeline.

    To add fine grained timings use:
    with inference_state.time(id="foo"):
        time.sleep(0.1)

    The timer object under the hood is shared among all operators

    To access the timings from timer_manager.measurements from Pipeline
    """

    # expected structured input and output types, to be defined by child classes
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None

    @classmethod
    def has_input_schema(cls) -> bool:
        """
        :return: True if this class has a defined pydantic input schema
        """
        if not cls.input_schema:
            return False

        return issubclass(cls.input_schema, BaseModel)

    @classmethod
    def has_output_schema(cls) -> bool:
        """
        :return: True if this class has a defined pydantic input schema
        """
        if not cls.output_schema:
            return False

        return issubclass(cls.output_schema, BaseModel)

    def __call__(
        self,
        *args,
        inference_state: InferenceState,
        **kwargs,
    ) -> Union[Any, Tuple[Any, Dict]]:
        """
        Parses inputs to this Operator and runs the run() method of this operator

        :param args: an unnamed arg may only be provided if it is of the type of the
            input_schema
        :param inference_state: inference_state for the pipeline.
        :param pipeline_state: pipeline_state for the pipeline. The values in the state
            are created during pipeline creation and are read-only during inference.
        :param kwargs: kwargs when not initializing from an instantiated schema
        :return: operator output. Some operators may return a tuple, consisting of the
            output and the state update in the form of a dictionary
        """
        if self.has_input_schema():
            if len(args) > 1:
                raise ValueError(
                    f"The operator requires an {self.input_schema}. Too many arguments"
                    "provided."
                )
            elif args and isinstance(args[0], self.input_schema):
                inference_input = args[0]
            elif kwargs:
                inference_input = self.input_schema(**kwargs)
            else:
                raise ValueError(
                    "Can't resolve inputs. The values for the schema must be provided"
                    "in the form of a dictionary or an instance of the input_schema"
                    "object"
                )
            run_output = self.run(
                inference_input,
                inference_state=inference_state,
            )
        else:
            run_output = self.run(
                *args,
                inference_state=inference_state,
                **kwargs,
            )

        state_update = None
        if isinstance(run_output, tuple):
            run_output, state_update = run_output

        if self.has_output_schema():
            run_output = self.output_schema(**run_output)

        if state_update:
            return run_output, state_update
        return run_output

    @classmethod
    def create(
        cls,
        task: str,
        **kwargs,
    ) -> "Operator":
        """
        :param task: Operator task
        :param kwargs: extra task specific kwargs to be passed to task Operator
            implementation
        :return: operator object initialized for the given task
        """
        try:
            operator_constructor = OperatorRegistry.get_task_constructor(task)
        except Exception:
            raise
        return operator_constructor(**kwargs)

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        :return: result of this operator as the defined output schema if applicable
        """
        raise NotImplementedError

    def can_operate(self, inp: Any) -> bool:
        """
        Whether or not the given operator can run, based on input
        """
        return True

    def yaml(self):
        pass

    def json(self):
        pass
