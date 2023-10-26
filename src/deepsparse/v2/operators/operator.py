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
from typing import Optional, Type

from pydantic import BaseModel

from deepsparse.v2.utils import Context, OperatorSchema


__all__ = ["Operator"]


class Operator(ABC):
    """
    Base operator class - can represent any part of an ML pipeline
    """

    # expected structured input and output types, to be defined by child classes
    input_schema: Optional[Type[OperatorSchema]] = None
    output_schema: Optional[Type[OperatorSchema]] = None

    @abstractmethod
    def run(self, inp: OperatorSchema, context: Context) -> OperatorSchema:
        """
        :param inp: operator input, as the defined input schema if applicable
        :param context: pipeline context of already run operators
        :return: result of this operator as the defined output schema if applicable
        """
        raise NotImplementedError

    @classmethod
    def has_input_schema(cls) -> bool:
        """
        :return: True if this class has a defined pydantic input schema
        """
        return issubclass(cls.input_schema, BaseModel)

    @classmethod
    def has_output_schema(cls) -> bool:
        """
        :return: True if this class has a defined pydantic input schema
        """
        return issubclass(cls.output_schema, BaseModel)

    def __call__(
        self,
        *args,
        context: Optional[Context] = None,
        **kwargs,
    ) -> OperatorSchema:
        """
        Parses inputs to this Operator and runs the run() method of this operator

        :param args: an unnamed arg may only be provided
            if it is of the type of the input_schema
        :param context: pipeline context to pass to operator
        :param kwargs: kwargs when not initializing from an instantiated schema
        :return: operator output
        """
        if len(args) > 1:
            raise ValueError(
                f"Only 1 unnamed arg may be supplied to an Operator, found {len(args)}"
            )

        if len(args) == 1:
            if self.input_schema is not None and isinstance(args[0], self.input_schema):
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
        return self.run(inference_input, context=context)
