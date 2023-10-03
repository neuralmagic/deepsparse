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
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from deepsparse import Context
from deepsparse.loggers.base_logger import BaseLogger
from deepsparse.loggers.build_logger import logger_from_config
from deepsparse.loggers.constants import validate_identifier
from deepsparse.tasks import SupportedTasks, dynamic_import_task


__all__ = [
    "BasePipeline",
]

_REGISTERED_PIPELINES = {}


class BasePipeline(ABC):
    """
    Generic BasePipeline abstract class meant to wrap inference objects to include
    model-specific Pipelines objects. Any pipeline inherited from Pipeline objects
    should handle all model-specific input/output pre/post processing while BasePipeline
    is meant to serve as a generic wrapper. Inputs and outputs of BasePipelines should
    be serialized as pydantic Models.

    BasePipelines should not be instantiated by their constructors, but rather the
    `BasePipeline.create()` method. The task name given to `create` will be used to
    load the appropriate pipeline. The pipeline should inherit from `BasePipeline` and
    implement the `__call__`, `input_schema`, and `output_schema` abstract methods.

    Finally, the class definition should be decorated by the `BasePipeline.register`
    function. This defines the task name and task aliases for the pipeline and
    ensures that it will be accessible by `BasePipeline.create`. The implemented
    `BasePipeline` subclass must be imported at runtime to be accessible.

    Example:
    @BasePipeline.register(task="base_example")
    class BasePipelineExample(BasePipeline):
        def __init__(self, base_specific, **kwargs):
            self._base_specific = base_specific
            self.model_pipeline = Pipeline.create(task="..")
            super().__init__(**kwargs)
        # implementation of abstract methods

    :param alias: optional name to give this pipeline instance, useful when
    inferencing with multiple models. Default is None
    :param logger: An optional item that can be either a DeepSparse Logger object,
    or an object that can be transformed into one. Those object can be either
    a path to the logging config, or yaml string representation the logging
    config. If logger provided (in any form), the pipeline will log inference
    metrics to the logger. Default is None

    """

    def __init__(
        self,
        alias: Optional[str] = None,
        logger: Optional[Union[BaseLogger, str]] = None,
    ):

        self._alias = alias
        self.logger = (
            logger
            if isinstance(logger, BaseLogger)
            else (
                logger_from_config(
                    config=logger, pipeline_identifier=self._identifier()
                )
                if isinstance(logger, str)
                else None
            )
        )

    @abstractmethod
    def __call__(self, *args, **kwargs) -> BaseModel:
        """
        Runner function needed to stitch together any parsing, preprocessing, engine,
        and post-processing steps.

        :returns: pydantic model class that outputs of this pipeline must comply to
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

    @staticmethod
    def _get_task_constructor(task: str) -> Type["BasePipeline"]:
        """
        This function retrieves the class previously registered via
        `BasePipeline.register` or `Pipeline.register` for `task`.

        If `task` starts with "import:", it is treated as a module to be imported,
        and retrieves the task via the `TASK` attribute of the imported module.

        If `task` starts with "custom", then it is mapped to the "custom" task.

        :param task: The task name to get the constructor for
        :return: The class registered to `task`
        :raises ValueError: if `task` was not registered via `Pipeline.register`.
        """
        if task.startswith("import:"):
            # dynamically import the task from a file
            task = dynamic_import_task(module_or_path=task.replace("import:", ""))
        elif task.startswith("custom"):
            # support any task that has "custom" at the beginning via the "custom" task
            task = "custom"
        else:
            task = task.lower().replace("-", "_")

        # extra step to register pipelines for a given task domain
        # for cases where imports should only happen once a user specifies
        # that domain is to be used. (ie deepsparse.transformers will auto
        # install extra packages so should only import and register once a
        # transformers task is specified)
        SupportedTasks.check_register_task(task, _REGISTERED_PIPELINES.keys())

        if task not in _REGISTERED_PIPELINES:
            raise ValueError(
                f"Unknown Pipeline task {task}. Pipeline tasks should be "
                "must be declared with the Pipeline.register decorator. Currently "
                f"registered pipelines: {list(_REGISTERED_PIPELINES.keys())}"
            )

        return _REGISTERED_PIPELINES[task]

    @staticmethod
    def create(
        task: str,
        **kwargs,
    ) -> "BasePipeline":
        """
        :param task: name of task to create a pipeline for. Use "custom" for
            custom tasks (see `CustomTaskPipeline`).
        :param kwargs: extra task specific kwargs to be passed to task Pipeline
            implementation
        :return: pipeline object initialized for the given task
        """
        from deepsparse.pipeline import Bucketable, BucketingPipeline, Pipeline

        pipeline_constructor = BasePipeline._get_task_constructor(task)
        model_path = kwargs.get("model_path", None)

        if issubclass(pipeline_constructor, Pipeline):
            if (
                (model_path is None or model_path == "default")
                and hasattr(pipeline_constructor, "default_model_path")
                and pipeline_constructor.default_model_path
            ):
                model_path = pipeline_constructor.default_model_path

            if model_path is None:
                raise ValueError(
                    f"No model_path provided for pipeline {pipeline_constructor}. Must "
                    "provide a model path for pipelines that do not have a default "
                    "defined"
                )

            kwargs["model_path"] = model_path

        if issubclass(
            pipeline_constructor, Bucketable
        ) and pipeline_constructor.should_bucket(**kwargs):
            if kwargs.get("input_shapes", None):
                raise ValueError(
                    "Overriding input shapes not supported with Bucketing enabled"
                )
            if not kwargs.get("context", None):
                context = Context(
                    num_cores=kwargs.get("num_cores"),
                    num_streams=kwargs.get("num_streams"),
                )
                kwargs["context"] = context
            buckets = pipeline_constructor.create_pipeline_buckets(
                task=task,
                **kwargs,
            )
            return BucketingPipeline(pipelines=buckets)

        return pipeline_constructor(**kwargs)

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
        the pipeline through `BasePipeline.create()` or `Pipeline.create()`

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
                    f"task {task_name} already registered by BasePipeline.register. "
                    f"attempting to register pipeline: {pipeline_class}, but"
                    f"pipeline: {_REGISTERED_PIPELINES[task_name]}, already registered"
                )
            _REGISTERED_PIPELINES[task_name] = pipeline_class

        def _register_pipeline_tasks_decorator(pipeline_class: BasePipeline):
            if not issubclass(pipeline_class, cls):
                raise RuntimeError(
                    f"Attempting to register pipeline {pipeline_class}. "
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
    def from_config(
        cls,
        config: Union["PipelineConfig", str, Path],  # noqa: F821
        logger: Optional[BaseLogger] = None,
    ) -> "BasePipeline":
        """
        :param config: PipelineConfig object, filepath to a json serialized
            PipelineConfig, or raw string of a json serialized PipelineConfig
        :param logger: An optional DeepSparse Logger object for inference
            logging. Default is None
        :return: loaded Pipeline object from the config
        """
        from deepsparse.pipeline import PipelineConfig

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
            alias=config.alias,
            logger=logger,
            **config.kwargs,
        )

    @property
    def alias(self) -> str:
        """
        :return: optional name to give this pipeline instance, useful when
            inferencing with multiple models
        """
        return self._alias

    def to_config(self) -> "PipelineConfig":  # noqa: F821
        """
        :return: PipelineConfig that can be used to reload this object
        """
        from deepsparse.pipeline import PipelineConfig

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
            alias=self.alias,
            kwargs=kwargs,
        )

    def log(
        self,
        identifier: str,
        value: Any,
        category: str,
    ):
        """
        Pass the logged data to the DeepSparse logger object (if present).

        :param identifier: The string name assigned to the logged value
        :param value: The logged data structure
        :param category: The metric category that the log belongs to
        """
        if not self.logger:
            return

        identifier = f"{self._identifier()}/{identifier}"
        validate_identifier(identifier)
        self.logger.log(
            identifier=identifier,
            value=value,
            category=category,
            pipeline_name=self._identifier(),
        )
        return

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

    def _identifier(self):
        # get pipeline identifier; used in the context of logging
        if not hasattr(self, "task"):
            self.task = None
        return f"{self.alias or self.task or 'unknown_pipeline'}"
