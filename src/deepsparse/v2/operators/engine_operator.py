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

from typing import Any, Dict, List, Optional, Union

from deepsparse import Context, Engine, MultiModelEngine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.utils import join_engine_outputs, split_engine_inputs
from deepsparse.v2.operators import Operator


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]

__all__ = ["EngineOperator"]


class EngineOperator(Operator):
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

        self.engine = self._create_engine(model_path, engine_type, engine_args)

    def _create_engine(
        self, onnx_file_path: str, engine_type: str, engine_args: Dict
    ) -> Union[Engine, MultiModelEngine, ORTEngine]:
        """
        Create an inference engine for a given ONNX model

        :param onnx_file_path: path to ONNX model file
        :param engine_type: type of engine to create.
        :param engine_args: arguments to pass to engine constructor
        :param context: context to use for engine
        :return: inference engine
        """
        engine_type = engine_type.lower()

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

    def run(self, inp: Any, context: Optional[Context]) -> Any:
        batches, orig_batch_size = self.expand_inputs(engine_inputs=inp)
        batches_outputs = list(map(self.engine, batches))
        engine_outputs = self.condense_inputs(
            batch_outputs=batches_outputs, orig_batch_size=orig_batch_size
        )
        return engine_outputs

    def expand_inputs(self, **kwargs):
        return split_engine_inputs(kwargs["engine_inputs"], self._batch_size)

    def condense_inputs(self, **kwargs):
        batch_outputs = kwargs["batch_outputs"]
        orig_batch_size = kwargs["orig_batch_size"]
        return join_engine_outputs(batch_outputs, orig_batch_size)
