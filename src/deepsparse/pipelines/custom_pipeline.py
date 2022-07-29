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

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel

from deepsparse.pipeline import Pipeline
from deepsparse.utils.onnx import model_to_path


@Pipeline.register(task="custom")
class CustomTaskPipeline(Pipeline):
    """
    A utility class provided to make specifying custom pipelines easier.
    Instead of creating a subclass of Pipeline, you can instantiate this directly
    by passing in functions to call for pre and post processing.

    The easiest way to use this class is to just pass in the model path, which
    lets use directly interact with engine inputs/outputs:
    ```python
    pipeline = CustomPipeline(model_path="...")
    ```

    Alternatively, you can pass the pre/post processing functions into
    the constructor:
    ```python
    def yolo_preprocess(inputs: YOLOInput) -> List[np.ndarray]:
        ...

    def yolo_postprocess(engine_outputs: List[np.ndarray]) -> YOLOOutput:
        ...

    yolo = CustomPipeline(
        model_path="...",
        input_schema=YOLOInput,
        output_schema=YOLOOutput,
        process_inputs_fn=yolo_preprocess,
        process_outputs_fn=yolo_postprocess,
    )
    ```

    :param model_path: path on local system or SparseZoo stub to load the model from.
        Passed to :class:`Pipeline`.
    :param input_schema: Optional pydantic schema that describes the input to
        `process_inputs_fn`. If None, then raw data is passed to `process_inputs`.
    :param output_schema: Optional pydantic schema that describes the output from
        `process_outputs_fn`. If None, then raw engine outputs are returned from
        the pipeline.
    :param process_inputs_fn: Optional callable (function, method, lambda, etc) that
        mapsan `InputSchema` object to a list of numpy arrays that can be directly
        passed into the forward pass of the pipeline engine. If `None`, raw data is
        passed to the engine.
    :param process_outputs_fn: Optional callable (function, method, lambda, etc) that
        maps the list of numpy arrays that are the output of the engine forward pass
        into an `OutputSchema` object. If `None`, engine outputs are directly returned.
    """

    def __init__(
        self,
        model_path: str,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        process_inputs_fn: Optional[
            Callable[
                [BaseModel],
                Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]],
            ]
        ] = None,
        process_outputs_fn: Optional[Callable[[List[numpy.ndarray]], BaseModel]] = None,
        *args,
        **kwargs,
    ):
        if input_schema is None:
            input_schema = object
        elif not issubclass(input_schema, BaseModel):
            raise ValueError(
                f"input_schema must subclass BaseModel. Found {input_schema}"
            )

        if output_schema is None:
            output_schema = object
        elif not issubclass(output_schema, BaseModel):
            raise ValueError(
                f"output_schema must subclass BaseModel. Found {output_schema}"
            )

        if process_inputs_fn is None:
            process_inputs_fn = _passthrough

        if process_outputs_fn is None:
            process_outputs_fn = _passthrough

        self._input_schema = input_schema
        self._output_schema = output_schema
        self._process_inputs_fn = process_inputs_fn
        self._process_outputs_fn = process_outputs_fn
        super().__init__(model_path, *args, **kwargs)

    def setup_onnx_file_path(self) -> str:
        """
        :return: output from `model_to_path` using the `model_path`
            from the constructor
        """
        return model_to_path(self.model_path)

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: The `input_schema` from the constructor.
        """
        return self._input_schema

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: The `output_schema` from the constructor.
        """
        return self._output_schema

    def process_inputs(
        self, inputs: BaseModel
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        """
        :return: The output from calling `process_inputs_fn` from the constructor
            on the `inputs`.
        """
        return self._process_inputs_fn(inputs)

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        """
        :return: The output from calling `process_outputs_fn` from the constructor
            on the `engine_outputs`.
        """
        return self._process_outputs_fn(engine_outputs, **kwargs)


def _passthrough(x, **kwargs):
    return x
