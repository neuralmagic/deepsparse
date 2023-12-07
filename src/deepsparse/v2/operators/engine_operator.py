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

from copy import deepcopy
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from deepsparse import Context as EngineContext
from deepsparse import Engine, MultiModelEngine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.utils import join_engine_outputs, model_to_path, split_engine_inputs
from deepsparse.v2.operators import Operator


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]

__all__ = ["EngineOperator", "EngineOperatorInputs", "EngineOperatorOutputs"]


class EngineOperatorInputs(BaseModel):
    engine_inputs: List = Field(description="engine_inputs")
    engine: Optional[Union[ORTEngine, Engine]] = Field(
        description="override the engine to run forward pass with",
        default=None,
    )

    @classmethod
    def join(cls, inputs: List["EngineOperatorInputs"]) -> "EngineOperatorInputs":
        """
        :param inputs: list of separate EngineOperatorInputs, batch size must be 1
        :return: list of inputs joined into a single input with a multi batch size
        """
        all_engine_inputs = [engine_input.engine_inputs for engine_input in inputs]

        for engine_inputs in all_engine_inputs:
            if engine_inputs[0].shape[0] != 1:
                raise RuntimeError(
                    "join requires all inputs to have batch size 1, found input with "
                    f"batch size {engine_inputs[0].shape[0]}"
                )

        # use join_engine_outputs since dtype is the same
        joined_engine_inputs = join_engine_outputs(
            all_engine_inputs, len(all_engine_inputs)
        )

        return cls(engine_inputs=joined_engine_inputs)

    class Config:
        arbitrary_types_allowed = True


class EngineOperatorOutputs(BaseModel):
    engine_outputs: List = Field(description="engine outputs")

    def split(self) -> List["EngineOperatorOutputs"]:
        """
        :return: list of the current outputs split to a batch size of 1 each
        """
        # using split_engine_inputs since input/output dtypes
        # are the same (List[ndarray])
        split_outputs, _ = split_engine_inputs(self.engine_outputs, batch_size=1)

        return [self.__class__(engine_outputs=outputs) for outputs in split_outputs]


class EngineOperator(Operator):
    input_schema = EngineOperatorInputs
    output_schema = EngineOperatorOutputs

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        num_cores: int = None,
        num_streams: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        engine_context: Optional[EngineContext] = None,
        engine_kwargs: Dict = None,
    ):
        self.model_path = model_to_path(model_path)
        self.engine_context = engine_context
        self._batch_size = 1

        if self.engine_context is not None:
            num_cores = num_cores or self.engine_context.num_cores
            if self.engine_context.num_cores != num_cores:
                raise ValueError(
                    f"num_cores mismatch. Expected {self.engine_context.num_cores} "
                    f"from passed context, but got {num_cores} while "
                    f"instantiating Pipeline"
                )

        engine_args = dict(
            batch_size=self._batch_size,
            num_cores=num_cores,
            input_shapes=input_shapes,
        )
        if engine_type.lower() == DEEPSPARSE_ENGINE:
            engine_args["scheduler"] = scheduler
            engine_args["num_streams"] = num_streams

        self._engine_args = engine_args
        self._engine_type = engine_type

        if not engine_kwargs:
            engine_kwargs = {}

        self.engine = self.create_engine(**engine_kwargs)

    @property
    def batch_size(self) -> int:
        """
        :return: the batch size this engine operator is compiled at
        """
        return self._batch_size

    # TODO: maybe add a few args to make this less opaque?
    def create_engine(
        self,
        **kwargs,
    ) -> Union[Engine, MultiModelEngine, ORTEngine]:
        """
        Create an inference engine for a given ONNX model

        :param kwargs: overrides to engine_args used as kwargs for engine
            constructor/compilation
        :return: inference engine
        """

        onnx_file_path = kwargs.pop("model_path", self.model_path)
        engine_args = deepcopy(self._engine_args)
        engine_args.update(kwargs)
        engine_type = self._engine_type.lower()

        if engine_type == DEEPSPARSE_ENGINE:
            if self.engine_context is not None and isinstance(
                self.engine_context, EngineContext
            ):
                engine_args.pop("num_cores", None)
                engine_args.pop("scheduler", None)
                engine_args.pop("num_streams", None)
                engine_args["context"] = self.engine_context
                return MultiModelEngine(
                    model=onnx_file_path,
                    **engine_args,
                )
            engine_args.pop("cache_output_bools", None)
            return Engine(onnx_file_path, **engine_args)

        if engine_type == ORT_ENGINE:
            return ORTEngine(onnx_file_path, **engine_args)

        raise ValueError(
            f"Unknown engine_type {engine_type}. Supported values include: "
            f"{SUPPORTED_PIPELINE_ENGINES}"
        )

    def run(self, inp: EngineOperatorInputs, **kwargs) -> Dict:
        if inp.engine:
            # run with custom engine, do not split/join since custom engine
            # may run at any batch size, returning here as code below has a
            # planned refactor
            engine_outputs = inp.engine(inp.engine_inputs)
            return {"engine_outputs": engine_outputs}

        engine_outputs = self.engine(inp.engine_inputs)
        return {"engine_outputs": engine_outputs}
