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

from deepsparse import Context, Engine, MultiModelEngine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.utils import join_engine_outputs, model_to_path, split_engine_inputs
from deepsparse.v2.operators import Operator


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]

__all__ = ["EngineOperator"]


class EngineOperatorInputs(BaseModel):
    engine_inputs: List = Field(description="engine_inputs")
    _engine: Optional[Engine] = Field(
        description="override the engine to run forward pass with"
    )


class EngineOperatorOutputs(BaseModel):
    engine_outputs: List = Field(description="engine outputs")


class EngineOperator(Operator):
    input_schema = EngineOperatorInputs
    output_schema = EngineOperatorOutputs

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: Optional[int] = 1,
        num_cores: int = None,
        num_streams: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        engine_context: Optional[Context] = None,
    ):

        self._batch_size = batch_size
        self.model_path = model_to_path(model_path)
        self.engine_context = engine_context

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

        self.engine = self.create_engine()

    @property
    def batch_size(self) -> int:
        """
        :return: the batch size this engine operator is compiled at
        """
        return self._batch_size

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
        onnx_file_path = self.model_path
        engine_args = deepcopy(self._engine_args)
        engine_args.update(kwargs)
        engine_type = self._engine_type.lower()

        if engine_type == DEEPSPARSE_ENGINE:
            if self.engine_context is not None and isinstance(
                self.engine_context, Context
            ):
                engine_args.pop("num_cores", None)
                engine_args.pop("scheduler", None)
                engine_args.pop("num_streams", None)
                engine_args["context"] = self.engien_context
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

    def run(self, inp: EngineOperatorInputs) -> Dict:
        inp = inp.engine_inputs
        if inp._engine:
            # run with custom engine, do not split/join since custom engine
            # may run at any batch size, returning here as code below has a
            # planned refactor
            engine_outputs = inp._engine(inp)
            return {"engine_outputs": engine_outputs}
        batches, orig_batch_size = self.expand_inputs(engine_inputs=inp)
        batches_outputs = list(map(self.engine, batches))
        engine_outputs = self.condense_inputs(
            batch_outputs=batches_outputs, orig_batch_size=orig_batch_size
        )
        return {"engine_outputs": engine_outputs}

    def expand_inputs(self, **kwargs):
        return split_engine_inputs(kwargs["engine_inputs"], self._batch_size)

    def condense_inputs(self, **kwargs):
        batch_outputs = kwargs["batch_outputs"]
        orig_batch_size = kwargs["orig_batch_size"]
        return join_engine_outputs(batch_outputs, orig_batch_size)
