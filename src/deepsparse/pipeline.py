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

"""
Classes and registry for end to end inference pipelines that wrap an underlying
inference engine and include pre/postprocessing
"""


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy
from pydantic import BaseModel

from deepsparse import Engine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.tasks import SupportedTasks


__all__ = [
    "DEEPSPARSE_ENGINE",
    "ORT_ENGINE",
    "SUPPORTED_PIPELINE_ENGINES",
    "Pipeline",
]


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


_REGISTERED_PIPELINES = {}


class Pipeline(ABC):
    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
    ):
        self._model_path_orig = model_path
        self._model_path = model_path
        self._engine_type = engine_type

        self._engine_args = dict(
            batch_size=batch_size,
            num_cores=num_cores,
            input_shapes=input_shapes,
        )
        if engine_type.lower() == DEEPSPARSE_ENGINE:
            self._engine_args["scheduler"] = scheduler

        self._onnx_file_path = self.setup_onnx_file_path()
        self._engine = self.initialize_engine()
        pass

    def __call__(self, pipeline_inputs: BaseModel = None, **kwargs) -> BaseModel:
        if pipeline_inputs is None and kwargs:
            # parse kwarg inputs into the expected input format
            pipeline_inputs = self.input_model(**kwargs)

        # validate inputs format
        if not isinstance(pipeline_inputs, self.input_model):
            raise ValueError(
                f"Calling {self.__class__} requires passing inputs as an "
                f"{self.input_model} object or a list of kwargs used to create "
                f"a {self.input_model} object"
            )

        # run pipeline
        engine_inputs: List[numpy.ndarray] = self.process_inputs(pipeline_inputs)
        engine_outputs: List[numpy.ndarray] = self.engine(engine_inputs)
        pipeline_outputs = self.process_engine_outputs(engine_outputs)

        # validate outputs format
        if not isinstance(pipeline_outputs, self.output_model):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of {self.output_model}"
                f" found output of type {type(pipeline_outputs)}"
            )

        return pipeline_outputs

    @staticmethod
    def create(
        task: str,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        **kwargs,
    ):
        task = task.lower().replace("-", "_")

        # extra step to register pipelines for a given task domain
        # for cases where imports should only happen once a user specifies
        # that domain is to be used. (ie deepsparse.transformers will auto
        # install extra packages so should only import and register once a
        # transformers task is specified)
        SupportedTasks.check_register_task(task)

        if task not in _REGISTERED_PIPELINES:
            raise ValueError(
                f"Unknown Pipeline task {task}. Pipeline tasks should be "
                "must be declared with the Pipeline.register decorator. Currently "
                f"registered pipelines: {list(_REGISTERED_PIPELINES.keys())}"
            )

        return _REGISTERED_PIPELINES[task](
            model_path=model_path,
            engine_type=engine_type,
            batch_size=batch_size,
            num_cores=num_cores,
            scheduler=scheduler,
            input_shapes=input_shapes,
            **kwargs,
        )

    @classmethod
    def register(cls, task: str, task_aliases: Optional[List[str]]):
        task_names = [task]
        if task_aliases:
            task_names.extend(task_aliases)

        def _register_task(task_name, pipeline_class):
            if task_name in _REGISTERED_PIPELINES and (
                pipeline_class is not _REGISTERED_PIPELINES[task_name]
            ):
                raise RuntimeError(
                    f"task {task_name} already registered by Pipeline.register. "
                    f"attempting to register pipeline: {pipeline_class}, but"
                    f"pipeline: {_REGISTERED_PIPELINES[task_name]}, already registered"
                )
            _REGISTERED_PIPELINES[task_name] = pipeline_class

        def decorator(pipeline_class: Pipeline):
            if not issubclass(pipeline_class, cls):
                raise RuntimeError(
                    f"Attempting to register pipeline pipeline_class. "
                    f"Registered pipelines must inherit from {cls}"
                )
            for task_name in task_names:
                _register_task(task_name, pipeline_class)

            # set task and task_aliases as class level property
            pipeline_class.task = task
            pipeline_class.task_aliases = task_aliases

        return decorator

    @abstractmethod
    def setup_onnx_file_path(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, inputs: BaseModel) -> List[numpy.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def process_engine_outputs(self, engine_outputs: List[numpy.ndarray]) -> BaseModel:
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_model(self) -> BaseModel:
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_model(self) -> BaseModel:
        raise NotImplementedError()

    @property
    def model_path_orig(self) -> str:
        return self._model_path_orig

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def engine(self) -> Union[Engine, ORTEngine]:
        return self._engine

    @property
    def engine_args(self) -> Dict[str, Any]:
        return self._engine_args

    @property
    def engine_type(self) -> str:
        return self._engine_type

    @property
    def onnx_file_path(self) -> str:
        return self._onnx_file_path

    def initialize_engine(self) -> Union[Engine, ORTEngine]:
        engine_type = self.engine_type.lower()

        if engine_type == DEEPSPARSE_ENGINE:
            return Engine(self.onnx_file_path, **self._engine_args)
        elif engine_type == ORT_ENGINE:
            return ORTEngine(self.onnx_file_path, **self._engine_args)
        else:
            raise ValueError(
                f"Unknown engine_type {self.engine_type}. Supported values include: "
                f"{SUPPORTED_PIPELINE_ENGINES}"
            )
