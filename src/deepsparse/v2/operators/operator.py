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
from typing import Any, Optional, Type

from pydantic import BaseModel

from deepsparse.v2.utils import Context, InferenceState, PipelineState


__all__ = ["Operator"]


class Operator(ABC):
    """
    Base operator class - an operator should be defined for each functional part of the
    pipeline.
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
        context: Context,
        pipeline_state: PipelineState,
        inference_state: InferenceState,
        **kwargs,
    ) -> Any:
        """
        Parses inputs to this Operator and runs the run() method of this operator

        :param args: an unnamed arg may only be provided if it is of the type of the
            input_schema
        :param context: pipeline context to pass to operator
        :param kwargs: kwargs when not initializing from an instantiated schema
        :return: operator output
        """
        if len(args) > 1:
            raise ValueError(
                f"Only 1 unnamed arg may be supplied to an Operator, found {len(args)}"
            )

        if len(args) == 1:
            if self.input_schema is None:
                inference_input = args[0]
            elif self.input_schema is not None and isinstance(
                args[0], self.input_schema
            ):
                inference_input = args[0]
            else:
                raise ValueError(
                    f"1 arg supplied to Operator {self.__class__.__name__} but was not "
                    f"of expected type {self.input_schema}, found {type(args[0])}"
                )
        elif self.has_input_schema():
            inference_input = self.input_schema(**kwargs)
        else:
            inference_input = kwargs

        run_output, state_update = self.run(
            inp=inference_input,
            context=context,
            pipeline_state=pipeline_state,
            inference_state=inference_state,
        )
        if self.has_output_schema() and not isinstance(run_output, self.output_schema):
            return self.output_schema(**run_output), state_update
        return run_output, state_update

    @abstractmethod
    def run(
        self,
        inp: Any,
        context: Optional[Context],
        pipeline_state: PipelineState,
        inference_state: InferenceState,
    ) -> Any:
        """
        :param inp: operator input, as the defined input schema if applicable
        :param context: pipeline context of already run operators
        :return: result of this operator as the defined output schema if applicable
        """
        raise NotImplementedError

    def can_operate(
        self, inp: Any, context: Context, inference_state: InferenceState
    ) -> bool:
        """
        Whether or not the given operator can run, based on input, context, or state
        """
        raise NotImplementedError

    def expand_inputs(self, **kwargs):
        """
        Generic function to handle expanding values.
        """
        raise NotImplementedError

    def condense_inputs(self, **kwargs):
        """
        Generic function to handle condensing values.
        """
        raise NotImplementedError

    def yaml(self):
        pass

    def json(self):
        pass
