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


import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Engine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.tasks import SupportedTasks


__all__ = [
    "DEEPSPARSE_ENGINE",
    "ORT_ENGINE",
    "SUPPORTED_PIPELINE_ENGINES",
    "Pipeline",
    "PipelineConfig",
]


DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


_REGISTERED_PIPELINES = {}


class Pipeline(ABC):
    """
    Generic Pipeline abstract class meant to wrap inference engine objects to include
    data pre/post-processing. Inputs and outputs of pipelines should be serialized
    as pydantic Models.

    Pipelines should not be instantiated by their constructors, but rather the
    `Pipeline.create()` method. The task name given to `create` will be used to
    load the appropriate pipeline. When creating a Pipeline, the pipeline should
    inherit from `Pipeline` and implement the `setup_onnx_file_path`, `process_inputs`,
    `process_engine_outputs`, `input_schema`, and `output_schema` abstract methods.

    Finally, the class definition should be decorated by the `Pipeline.register`
    function. This defines the task name and task aliases for the pipeline and
    ensures that it will be accessible by `Pipeline.create`. The implemented
    `Pipeline` subclass must be imported at runtime to be accessible.

    Pipeline lifecycle:
     - On instantiation
         * `onnx_file_path` <- `setup_onnx_file_path`
         * `engine` <- `_initialize_engine`

     - on __call__:
         * `parsed_inputs: input_schema` <- `parse_inputs(*args, **kwargs)`
         * `pre_processed_inputs` <- `process_inputs(parsed_inputs)`
         * `engine_outputs` <- `engine(pre_processed_inputs)`
         * `outputs: output_schema` <- `process_engine_outputs(engine_outputs)`

    Example use of register:
     ```python
     @Pipeline.register(
     task="example_task",
     task_aliases=["example_alias_1", "example_alias_2"],
     )
     class PipelineImplementation(Pipeline):
     # implementation of Pipeline abstract methods here
     ```

    Example use of pipeline:
     ```python
     example_pipeline = Pipeline.create(
         task="example_task",
         model_path="model.onnx",
     )
     pipeline_outputs = example_pipeline(pipeline_inputs)
     ```

    :param model_path: path on local system or SparseZoo stub to load the model from
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: static batch size to use for inference. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    """

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
    ):
        self._model_path_orig = model_path
        self._model_path = model_path
        self._engine_type = engine_type
        self._alias = alias

        self._engine_args = dict(
            batch_size=batch_size,
            num_cores=num_cores,
            input_shapes=input_shapes,
        )
        if engine_type.lower() == DEEPSPARSE_ENGINE:
            self._engine_args["scheduler"] = scheduler

        self.onnx_file_path = self.setup_onnx_file_path()
        self.engine = self._initialize_engine()

    def __call__(self, *args, **kwargs) -> BaseModel:
        if "engine_inputs" in kwargs:
            raise ValueError(
                "invalid kwarg engine_inputs. engine inputs determined "
                f"by {self.__class__.__qualname__}.parse_inputs"
            )

        # parse inputs into input_schema schema if necessary
        pipeline_inputs = self.parse_inputs(*args, **kwargs)
        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )

        # run pipeline
        engine_inputs: List[numpy.ndarray] = self.process_inputs(pipeline_inputs)

        if isinstance(engine_inputs, tuple):
            engine_inputs, postprocess_kwargs = engine_inputs
        else:
            postprocess_kwargs = {}

        engine_outputs: List[numpy.ndarray] = self.engine_forward(engine_inputs)
        pipeline_outputs = self.process_engine_outputs(
            engine_outputs, **postprocess_kwargs
        )

        # validate outputs format
        if not isinstance(pipeline_outputs, self.output_schema):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of "
                f"{self.output_schema} found output of type {type(pipeline_outputs)}"
            )

        return pipeline_outputs

    @staticmethod
    def create(
        task: str,
        model_path: str = None,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        **kwargs,
    ) -> "Pipeline":
        """
        :param task: name of task to create a pipeline for
        :param model_path: path on local system or SparseZoo stub to load the model
            from. Some tasks may have a default model path
        :param engine_type: inference engine to use. Currently supported values
            include 'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
        :param batch_size: static batch size to use for inference. Default is 1
        :param num_cores: number of CPU cores to allocate for inference engine. None
            specifies all available cores. Default is None
        :param scheduler: (deepsparse only) kind of scheduler to execute with.
            Pass None for the default
        :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
            to use model as-is. Default is None
        :param alias: optional name to give this pipeline instance, useful when
            inferencing with multiple models. Default is None
        :param kwargs: extra task specific kwargs to be passed to task Pipeline
            implementation
        :return: pipeline object initialized for the given task
        """
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

        pipeline_constructor = _REGISTERED_PIPELINES[task]

        if (
            model_path is None
            and hasattr(pipeline_constructor, "default_model_path")
            and pipeline_constructor.default_model_path
        ):
            model_path = pipeline_constructor.default_model_path

        if model_path is None:
            raise ValueError(
                f"No model_path provided for pipeline {pipeline_constructor}. Must "
                "provide a model path for pipelines that do not have a default defined"
            )

        return pipeline_constructor(
            model_path=model_path,
            engine_type=engine_type,
            batch_size=batch_size,
            num_cores=num_cores,
            scheduler=scheduler,
            input_shapes=input_shapes,
            alias=alias,
            **kwargs,
        )

    @classmethod
    def register(
        cls,
        task: str,
        task_aliases: Optional[List[str]] = None,
        default_model_path: Optional[str] = None,
    ):
        """
        Pipeline implementer class decorator that registers the pipeline
        task name and its aliases as valid tasks that can be used to load
        the pipeline through `Pipeline.create()`.

        Multiple pipelines may not have the same task name. An error will
        be raised if two different pipelines attempt to register the same task name

        :param task: main task name of this pipeline
        :param task_aliases: list of extra task names that may be used to reference
            this pipeline. Default is None
        :param default_model_path: path (ie zoo stub) to use as default for this
            task if None is provided
        """
        task_names = [task]
        if task_aliases:
            task_names.extend(task_aliases)

        task_names = [task_name.lower().replace("-", "_") for task_name in task_names]

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

        def _register_pipeline_tasks_decorator(pipeline_class: Pipeline):
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
            pipeline_class.default_model_path = default_model_path

            return pipeline_class

        return _register_pipeline_tasks_decorator

    @classmethod
    def from_config(cls, config: Union["PipelineConfig", str, Path]) -> "Pipeline":
        """
        :param config: PipelineConfig object, filepath to a json serialized
            PipelineConfig, or raw string of a json serialized PipelineConfig
        :return: loaded Pipeline object from the config
        """
        if isinstance(config, Path) or (
            isinstance(config, str) and os.path.exists(config)
        ):
            if isinstance(config, str):
                config = Path(config)
            config = PipelineConfig.parse_file(config)
        if isinstance(config, str):
            config = PipelineConfig.parse_raw(config)

        return cls.create(
            task=config.task,
            model_path=config.model_path,
            engine_type=config.engine_type,
            batch_size=config.batch_size,
            num_cores=config.num_cores,
            scheduler=config.scheduler,
            input_shapes=config.input_shapes,
            alias=config.alias,
            **config.kwargs,
        )

    @abstractmethod
    def setup_onnx_file_path(self) -> str:
        """
        Performs any setup to unwrap and process the given `model_path` and other
        class properties into an inference ready onnx file to be compiled by the
        engine of the pipeline

        :return: file path to the ONNX file for the engine to compile
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(
        self,
        inputs: BaseModel,
    ) -> Union[List[numpy.ndarray], Tuple[List[numpy.ndarray], Dict[str, Any]]]:
        """
        :param inputs: inputs to the pipeline. Must be the type of the `input_schema`
            of this pipeline
        :return: inputs of this model processed into a list of numpy arrays that
            can be directly passed into the forward pass of the pipeline engine. Can
            also include a tuple with engine inputs and special key word arguments
            to pass to process_engine_outputs to facilitate information from the raw
            inputs to postprocessing that may not be included in the engine inputs
        """
        raise NotImplementedError()

    @abstractmethod
    def process_engine_outputs(
        self,
        engine_outputs: List[numpy.ndarray],
        **kwargs,
    ) -> BaseModel:
        """
        :param engine_outputs: list of numpy arrays that are the output of the engine
            forward pass
        :return: outputs of engine post-processed into an object in the `output_schema`
            format of this pipeline
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        raise NotImplementedError()

    @property
    def alias(self) -> str:
        """
        :return: optional name to give this pipeline instance, useful when
            inferencing with multiple models
        """
        return self._alias

    @property
    def model_path_orig(self) -> str:
        """
        :return: value originally passed to the `model_path` argument to initialize
            this Pipeline
        """
        return self._model_path_orig

    @property
    def model_path(self) -> str:
        """
        :return: path on local system to the onnx file of this model or directory
            containing a model.onnx file along with supporting files
        """
        return self._model_path

    @property
    def engine_args(self) -> Dict[str, Any]:
        """
        :return: arguments besides onnx filepath used to instantiate engine
        """
        return self._engine_args

    @property
    def engine_type(self) -> str:
        """
        :return: type of inference engine used for model forward pass
        """
        return self._engine_type

    def to_config(self) -> "PipelineConfig":
        """
        :return: PipelineConfig that can be used to reload this object
        """

        if not hasattr(self, "task"):
            raise RuntimeError(
                f"{self.__class__} instance has no attribute task. Pipeline objects "
                "must have a task to be serialized to a config. Pipeline objects "
                "must be declared with the Pipeline.register object to be assigned a "
                "task"
            )

        # parse any additional properties as kwargs
        kwargs = {}
        for attr_name, attr in self.__class__.__dict__.items():
            if isinstance(attr, property) and attr_name not in dir(PipelineConfig):
                kwargs[attr_name] = getattr(self, attr_name)

        return PipelineConfig(
            task=self.task,
            model_path=self.model_path_orig,
            engine_type=self.engine_type,
            batch_size=self.batch_size,
            num_cores=self.num_cores,
            scheduler=self.scheduler,
            input_shapes=self.input_shapes,
            alias=self.alias,
            kwargs=kwargs,
        )

    def parse_inputs(self, *args, **kwargs) -> BaseModel:
        """
        :param args: ordered arguments to pipeline, only an input_schema object
            is supported as an arg for this function
        :param kwargs: keyword arguments to pipeline
        :return: pipeline arguments parsed into the given `input_schema`
            schema if necessary. If an instance of the `input_schema` is provided
            it will be returned
        """
        # passed input_schema schema directly
        if len(args) == 1 and isinstance(args[0], self.input_schema) and not kwargs:
            return args[0]

        if args:
            raise ValueError(
                f"pipeline {self.__class__} only supports either only a "
                f"{self.input_schema} object. or keyword arguments to be construct "
                f"one. Found {len(args)} args and {len(kwargs)} kwargs"
            )

        return self.input_schema(**kwargs)

    def engine_forward(self, engine_inputs: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :return: result of forward pass to Pipeline engine
        """
        return self.engine(engine_inputs)

    def _initialize_engine(self) -> Union[Engine, ORTEngine]:
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


class PipelineConfig(BaseModel):
    """
    Configuration for creating a Pipeline object

    Can be used to create a Pipeline from a config object or file with
    Pipeline.from_config(), or used as a building block for other configs
    such as for deepsparse.server
    """

    task: str = Field(
        description="name of task to create a pipeline for",
    )
    model_path: str = Field(
        description="path on local system or SparseZoo stub to load the model from",
    )
    engine_type: str = Field(
        default=DEEPSPARSE_ENGINE,
        description=(
            "inference engine to use. Currently supported values include "
            "'deepsparse' and 'onnxruntime'. Default is 'deepsparse'"
        ),
    )
    batch_size: int = Field(
        default=1,
        description=("static batch size to use for inference. Default is 1"),
    )
    num_cores: int = Field(
        default=None,
        description=(
            "number of CPU cores to allocate for inference engine. None"
            "specifies all available cores. Default is None"
        ),
    )
    scheduler: str = Field(
        default="async",
        description=(
            "(deepsparse only) kind of scheduler to execute with. Defaults to async"
        ),
    )
    input_shapes: List[List[int]] = Field(
        default=None,
        description=(
            "list of shapes to set ONNX the inputs to. Pass None to use model as-is. "
            "Default is None"
        ),
    )
    alias: str = Field(
        default=None,
        description=(
            "optional name to give this pipeline instance, useful when inferencing "
            "with multiple models. Default is None"
        ),
    )
    kwargs: Dict[str, Any] = Field(
        default={},
        description=(
            "Additional arguments for inference with the model that will be passed "
            "into the pipeline as kwargs"
        ),
    )
