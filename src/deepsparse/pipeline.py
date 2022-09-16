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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Context, Engine, MultiModelEngine, Scheduler
from deepsparse.benchmark import ORTEngine
from deepsparse.cpu import cpu_details
from deepsparse.tasks import SupportedTasks, dynamic_import_task
from deepsparse.timing import InferencePhases, InferenceTimingSchema, TimingBuilder


__all__ = [
    "DEEPSPARSE_ENGINE",
    "ORT_ENGINE",
    "SUPPORTED_PIPELINE_ENGINES",
    "Pipeline",
    "PipelineConfig",
    "question_answering_pipeline",
    "text_classification_pipeline",
    "zero_shot_text_classification_pipeline",
    "token_classification_pipeline",
    "image_classification_pipeline",
    "yolo_pipeline",
    "Bucketable",
    "BucketingPipeline",
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
    :param batch_size: static batch size to use for inference. None represents
        dynamic batch mode (Pipeline will accept any batch size). Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param context: Optional Context object to use for creating instances of
        MultiModelEngine. The Context contains a shared scheduler along with
        other runtime information that will be used across instances of the
        MultiModelEngine to provide optimal performance when running multiple
        models concurrently
    :param executor: An optional ThreadPoolExecutor() object, if provided the
        pipeline executes inference requests in a non-blocking manner and returns
        a Future object, call Future.result() on returned object to get the result.
        Can also accept an int number of workers, a ThreadPoolExecutor object is
        auto-initialized with the specified integer in that case; None represents
        synchronous execution - if running in dynamic batch mode a default
        ThreadPoolExecutor with default workers equal to the number of available
        cores / 2
    """

    def __init__(
        self,
        model_path: str,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: Optional[int] = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        context: Optional[Context] = None,
        executor: Optional[Union[ThreadPoolExecutor, int]] = None,
    ):
        self._model_path_orig = model_path
        self._model_path = model_path
        self._engine_type = engine_type
        self._batch_size = batch_size
        self._alias = alias
        self.context = context

        self.executor, self._num_async_workers = _initialize_executor_and_workers(
            batch_size=batch_size,
            workers_or_executor=executor,
        )

        if self.context is not None:
            num_cores = num_cores or self.context.num_cores
            if self.context.num_cores != num_cores:
                raise ValueError(
                    f"num_cores mismatch. Expected {self.context.num_cores} "
                    f"from passed context, but got {num_cores} while "
                    f"instantiating Pipeline"
                )

        self._engine_args = dict(
            batch_size=self._batch_size or 1,  # bs=1 for dynamic batch
            num_cores=num_cores,
            input_shapes=input_shapes,
        )
        if engine_type.lower() == DEEPSPARSE_ENGINE:
            self._engine_args["scheduler"] = scheduler

        self.onnx_file_path = self.setup_onnx_file_path()
        self.engine = self._initialize_engine()

        self._batch_size = self._batch_size or 1

    def __call__(self, *args, monitoring: bool = False, **kwargs) -> BaseModel:
        if "engine_inputs" in kwargs:
            raise ValueError(
                "invalid kwarg engine_inputs. engine inputs determined "
                f"by {self.__class__.__qualname__}.parse_inputs"
            )
        timer = TimingBuilder()

        # parse inputs into input_schema
        timer.start(InferencePhases.TOTAL_INFERENCE)
        timer.start(InferencePhases.PRE_PROCESS)
        pipeline_inputs = self.parse_inputs(*args, **kwargs)
        if not isinstance(pipeline_inputs, self.input_schema):
            raise RuntimeError(
                f"Unable to parse {self.__class__} inputs into a "
                f"{self.input_schema} object. Inputs parsed to {type(pipeline_inputs)}"
            )
        # batch size of the inputs may be `> self._batch_size` at this point
        engine_inputs: List[numpy.ndarray] = self.process_inputs(pipeline_inputs)
        if isinstance(engine_inputs, tuple):
            engine_inputs, postprocess_kwargs = engine_inputs
        else:
            postprocess_kwargs = {}
        timer.stop(InferencePhases.PRE_PROCESS)

        # split inputs into batches of size `self._batch_size`
        timer.start(InferencePhases.ENGINE_FORWARD)
        batches = self.split_engine_inputs(engine_inputs, self._batch_size)

        # submit split batches to engine threadpool
        batch_outputs = list(self.executor.map(self.engine_forward, batches))

        # join together the batches of size `self._batch_size`
        engine_outputs = self.join_engine_outputs(batch_outputs)
        timer.stop(InferencePhases.ENGINE_FORWARD)

        timer.start(InferencePhases.POST_PROCESS)
        pipeline_outputs = self.process_engine_outputs(
            engine_outputs, **postprocess_kwargs
        )
        if not isinstance(pipeline_outputs, self.output_schema):
            raise ValueError(
                f"Outputs of {self.__class__} must be instances of "
                f"{self.output_schema} found output of type {type(pipeline_outputs)}"
            )
        timer.stop(InferencePhases.POST_PROCESS)
        timer.stop(InferencePhases.TOTAL_INFERENCE)

        if not monitoring:
            return pipeline_outputs

        else:
            inference_timing = InferenceTimingSchema(**timer.build())
            return pipeline_outputs, pipeline_inputs, engine_inputs, inference_timing

    def run_with_monitoring(
        self, *args, **kwargs
    ) -> Tuple[BaseModel, BaseModel, Any, InferenceTimingSchema]:
        """
        Run the inference forward pass and additionally
        return extra monitoring information

        :return:
            pipeline_outputs: outputs from the inference pipeline
            pipeline_inputs: inputs to the inference pipeline
            engine_inputs: direct input to the inference engine
            inference_timing: BaseModel, that contains the information about time
                elapsed during the inference steps: pre-processing,
                engine-forward, post-processing, as well as
                the total elapsed time
        """
        (
            pipeline_outputs,
            pipeline_inputs,
            engine_inputs,
            inference_timing,
        ) = self.__call__(*args, monitoring=True, **kwargs)
        return pipeline_outputs, pipeline_inputs, engine_inputs, inference_timing

    @staticmethod
    def split_engine_inputs(
        items: List[numpy.ndarray], batch_size: int
    ) -> List[List[numpy.ndarray]]:
        """
        Splits each item into numpy arrays with the first dimension == `batch_size`.

        For example, if `items` has three numpy arrays with the following
        shapes: `[(4, 32, 32), (4, 64, 64), (4, 128, 128)]`

        Then with `batch_size==4` the output would be:
        ```
        [[(4, 32, 32), (4, 64, 64), (4, 128, 128)]]
        ```

        Then with `batch_size==2` the output would be:
        ```
        [
            [(2, 32, 32), (2, 64, 64), (2, 128, 128)],
            [(2, 32, 32), (2, 64, 64), (2, 128, 128)],
        ]
        ```

        Then with `batch_size==1` the output would be:
        ```
        [
            [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
            [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
            [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
            [(1, 32, 32), (1, 64, 64), (1, 128, 128)],
        ]
        ```
        """
        # if not all items here are numpy arrays, there's an internal
        # but in the processing code
        assert all(isinstance(item, numpy.ndarray) for item in items)

        # if not all items have the same batch size, there's an
        # internal bug in the processing code
        total_batch_size = items[0].shape[0]
        assert all(item.shape[0] == total_batch_size for item in items)

        if total_batch_size % batch_size != 0:
            raise RuntimeError(
                f"batch size of {total_batch_size} passed into pipeline "
                f"is not divisible by model batch size of {batch_size}"
            )

        batches = []
        for i_batch in range(total_batch_size // batch_size):
            start = i_batch * batch_size
            batches.append([item[start : start + batch_size] for item in items])
        return batches

    @staticmethod
    def join_engine_outputs(
        batch_outputs: List[List[numpy.ndarray]],
    ) -> List[numpy.ndarray]:
        """
        Joins list of engine outputs together into one list using `numpy.concatenate`.

        This is the opposite of `Pipeline.split_engine_inputs`.
        """
        return list(map(numpy.concatenate, zip(*batch_outputs)))

    @staticmethod
    def _get_task_constructor(task: str) -> Type["Pipeline"]:
        """
        This function retrieves the class previously registered via `Pipeline.register`
        for `task`.

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
        model_path: str = None,
        engine_type: str = DEEPSPARSE_ENGINE,
        batch_size: int = 1,
        num_cores: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        alias: Optional[str] = None,
        context: Optional[Context] = None,
        **kwargs,
    ) -> "Pipeline":
        """
        :param task: name of task to create a pipeline for. Use "custom" for
            custom tasks (see `CustomTaskPipeline`).
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
        :param context: Optional Context object to use for creating instances of
            MultiModelEngine. The Context contains a shared scheduler along with
            other runtime information that will be used across instances of the
            MultiModelEngine to provide optimal performance when running
            multiple models concurrently
        :param kwargs: extra task specific kwargs to be passed to task Pipeline
            implementation
        :return: pipeline object initialized for the given task
        """
        pipeline_constructor = Pipeline._get_task_constructor(task)

        if (
            (model_path is None or model_path == "default")
            and hasattr(pipeline_constructor, "default_model_path")
            and pipeline_constructor.default_model_path
        ):
            model_path = pipeline_constructor.default_model_path

        if model_path is None:
            raise ValueError(
                f"No model_path provided for pipeline {pipeline_constructor}. Must "
                "provide a model path for pipelines that do not have a default defined"
            )

        if issubclass(
            pipeline_constructor, Bucketable
        ) and pipeline_constructor.should_bucket(**kwargs):
            if input_shapes:
                raise ValueError(
                    "Overriding input shapes not supported with Bucketing enabled"
                )
            if not context:
                context = Context(num_cores=num_cores)
            buckets = pipeline_constructor.create_pipeline_buckets(
                task=task,
                model_path=model_path,
                engine_type=engine_type,
                batch_size=batch_size,
                alias=alias,
                context=context,
                **kwargs,
            )
            return BucketingPipeline(pipelines=buckets)

        return pipeline_constructor(
            model_path=model_path,
            engine_type=engine_type,
            batch_size=batch_size,
            num_cores=num_cores,
            scheduler=scheduler,
            input_shapes=input_shapes,
            alias=alias,
            context=context,
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
        config: Union["PipelineConfig", str, Path],
        context: Optional[Context] = None,
    ) -> "Pipeline":
        """
        :param config: PipelineConfig object, filepath to a json serialized
            PipelineConfig, or raw string of a json serialized PipelineConfig
        :param context: Optional Context object to use for creating instances of
            MultiModelEngine. The Context contains a shared scheduler along with
            other runtime information that will be used across instances of the
            MultiModelEngine to provide optimal performance when running
            multiple models concurrently
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
            context=context,
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
            if self.context is not None and isinstance(self.context, Context):
                self._engine_args.pop("num_cores", None)
                self._engine_args.pop("scheduler", None)
                self._engine_args["context"] = self.context
                return MultiModelEngine(
                    model=self.onnx_file_path,
                    **self._engine_args,
                )
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
    batch_size: Optional[int] = Field(
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


class BucketingPipeline(object):
    """
    A Proxy class that adds Bucketing functionality to Pipelines

    :param pipelines: A list of Pipeline objects/buckets that implement
        `Bucketable` contract
    """

    def __init__(self, pipelines: List[Pipeline]):
        if not (pipelines and isinstance(pipelines, list)):
            raise ValueError(
                "Expected a non empty List of pipeline objects but got " f"{pipelines}"
            )
        self._pipelines = pipelines
        self._pipeline_class = pipelines[0].__class__
        self._validate_pipeline_class()

    def __call__(self, *args, **kwargs):
        bucket, parsed_inputs = self._choose_bucket(*args, **kwargs)
        return bucket(parsed_inputs)

    def _choose_bucket(self, *args, **kwargs):
        parsed_inputs = self._pipelines[-1].parse_inputs(*args, **kwargs)
        bucket = self._pipeline_class.route_input_to_bucket(
            input_schema=parsed_inputs,
            pipelines=self._pipelines,
        )
        return bucket, parsed_inputs

    def __getattr__(self, item):
        value = getattr(self._pipelines[0].__class__, item)

        if isinstance(value, property):
            return getattr(self._pipelines[0], item)

        raise AttributeError(
            f"{item} not found in {self.__class__.__name__}, "
            f"and is not a property of {self._pipeline_class.__name__}"
        )

    @property
    def input_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that inputs to this pipeline must comply to
        """
        return self._pipelines[0].input_schema

    @property
    def output_schema(self) -> Type[BaseModel]:
        """
        :return: pydantic model class that outputs of this pipeline must comply to
        """
        return self._pipelines[0].output_schema

    def _validate_pipeline_class(self):
        # validate all pipelines belong to the same class

        if not issubclass(self._pipeline_class, Bucketable):
            raise ValueError(f"{self._pipeline_class} is not Bucketable")

        is_valid = all(
            isinstance(pipeline, self._pipeline_class) for pipeline in self._pipelines
        )

        if not is_valid:
            raise ValueError(
                "All Pipeline Buckets must belong to the same Pipeline Class"
            )


class Bucketable(ABC):
    """
    A contract, that ensures implementing Pipeline class can create multiple Pipeline
    instances and route each input sample to correct instance based off of specific
    implementations of abstract methods defined in this contract
    """

    @staticmethod
    @abstractmethod
    def should_bucket(*args, **kwargs) -> bool:
        """
        :returns: True if buckets should be created else False
        """
        pass

    @staticmethod
    @abstractmethod
    def create_pipeline_buckets(*args, **kwargs) -> List[Pipeline]:
        """
        :return: Create and return a list of Pipeline objects
            representing different buckets
        """
        pass

    @staticmethod
    @abstractmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        """
        :param input_schema: The schema representing an input to the pipeline
        :param pipelines: Different buckets to be used
        :return: The correct Pipeline object (or Bucket) to route input to
        """
        pass


def _initialize_executor_and_workers(
    batch_size: Optional[int],
    workers_or_executor: Optional[Union[int, ThreadPoolExecutor]],
) -> Tuple[Optional[ThreadPoolExecutor], int]:
    if isinstance(workers_or_executor, ThreadPoolExecutor):
        num_async_workers = workers_or_executor._max_workers  # noqa
        executor = workers_or_executor
    elif isinstance(workers_or_executor, int):
        num_async_workers = max(1, workers_or_executor)
        executor = ThreadPoolExecutor(max_workers=num_async_workers)
    elif batch_size is None and workers_or_executor is None:
        # default num workers to num available cores / 2
        num_cpu_cores_avaailable = cpu_details()[0]
        num_async_workers = max(1, num_cpu_cores_avaailable // 2)
        executor = ThreadPoolExecutor(max_workers=num_async_workers)
    elif workers_or_executor is not None:
        raise ValueError(
            "Expected an int or ThreadPoolExecutor to run in async mode"
            f" but got {workers_or_executor} of type {type(workers_or_executor)}"
        )
    else:
        executor = ThreadPoolExecutor(max_workers=1)
        num_async_workers = 1

    if batch_size is None and executor is None:
        raise ValueError(
            "Must have an ThreadPoolExecutor for running in dynamic batch mode "
            f"but got {None}"
        )

    return executor, num_async_workers


def question_answering_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers question_answering pipeline

    example instantiation:
    ```python
    question_answering = Pipeline.create(
        task="question_answering",
        model_path="question_answering_model_dir/",
    )
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param doc_stride: if the context is too long to fit with the question for the
        model, it will be split in several chunks with some overlap. This argument
        controls the size of that overlap. Currently, only reading the first span
        is supported (everything after doc_stride will be truncated). Default
        is 128
    :param max_question_len: maximum length of the question after tokenization.
        It will be truncated if needed. Default is 64
    :param max_answer_len: maximum length of answer after decoding. Default is 15
    """
    return Pipeline.create("question_answering", *args, **kwargs)


def text_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers text classification pipeline

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    example batch size 1, single text inputs (ie sentiment analysis):
    ```python
    sentiment = text_classifier("the food tastes great")
    sentiment = text_classifier(["the food tastes great"])
    sentiment = text_classifier([["the food tastes great"]])
    ```

    example batch size 1, multi text input (ie QQP like tasks):
    ```python
    prediction = text_classifier([["how is the food?", "what is the food?"]])
    ```

    example batch size n, single text inputs:
    ```python
    sentiments = text_classifier(["the food tastes great", "the food tastes bad"])
    sentiments = text_classifier([["the food tastes great"], ["the food tastes bad"]])
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param return_all_scores: if True, instead of returning the prediction as the
        argmax of model class predictions, will return all scores and labels as
        a list for each result in the batch. Default is False
    """
    return Pipeline.create("text_classification", *args, **kwargs)


def sentiment_analysis_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers text classification pipeline

    example instantiation:
    ```python
    text_classifier = Pipeline.create(
        task="text_classification",
        model_path="text_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    example batch size 1, single text inputs (ie sentiment analysis):
    ```python
    sentiment = text_classifier("the food tastes great")
    sentiment = text_classifier(["the food tastes great"])
    sentiment = text_classifier([["the food tastes great"]])
    ```

    example batch size 1, multi text input (ie QQP like tasks):
    ```python
    prediction = text_classifier([["how is the food?", "what is the food?"]])
    ```

    example batch size n, single text inputs:
    ```python
    sentiments = text_classifier(["the food tastes great", "the food tastes bad"])
    sentiments = text_classifier([["the food tastes great"], ["the food tastes bad"]])
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param return_all_scores: if True, instead of returning the prediction as the
        argmax of model class predictions, will return all scores and labels as
        a list for each result in the batch. Default is False
    """
    return Pipeline.create("text_classification", *args, **kwargs)


def token_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers token classification pipeline

    example instantiation:
    ```python
    token_classifier = Pipeline.create(
        task="token_classification",
        model_path="token_classification_model_dir/",
        batch_size=BATCH_SIZE,
    )
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param aggregation_strategy: how to aggregate tokens in postprocessing. Options
        include 'none', 'simple', 'first', 'average', and 'max'. Default is None
    :param ignore_labels: list of label names to ignore in output. Default is
        ['0'] which ignores the default known class label
    """
    return Pipeline.create("token_classification", *args, **kwargs)


def image_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Image classification pipeline for DeepSparse

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
    :param class_names: Optional dict, or json file of class names to use for
        mapping class ids to class labels. Default is None
    """
    return Pipeline.create("image_classification", *args, **kwargs)


def yolo_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Image Segmentation YOLO pipeline for DeepSparse

    :param model_path: path on local system or SparseZoo stub to load the model from
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
    :param class_names: Optional string identifier, dict, or json file of
        class names to use for mapping class ids to class labels. Default is
        `coco`
    """
    return Pipeline.create("yolo", *args, **kwargs)


def haystack_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Neural Magic pipeline for running Haystack DocumentSearchPipeline.
    Supports selected Haystack Nodes as well as Haystack nodes integrated
    with the Neural Magic DeepSparse Engine

    example embedding model instantiation:
    ```python
    haystack_pipeline = Pipeline.create(
        task="information_retrieval_haystack",
        model_path="masked_language_modeling_model_dir/",
        config={
            "document_store": "InMemoryDocumentStore",
            "document_store_args": {
                "similarity": "cosine",
                "use_gpu": False,
            },
            "retriever": "DeepSparseEmbeddingRetriever",
            "retriever_args": {
                "extraction_strategy": "reduce_mean"
            }
        },
    )
    ```

    example deepsparse biencoder instantiation
    ```python
    haystack_pipeline = Pipeline.create(
        task="information_retrieval_haystack",
        config={
            "document_store": "InMemoryDocumentStore",
            "document_store_args": {
                "similarity": "cosine",
                "use_gpu": False,
            },
            "retriever": "DeepSparseDensePassageRetriever",
            "retriever_args": {
                "query_model_path": "./query_model",
                "passage_model_path": "./passage_model"
            }
        },
    )
    ```

    writing documents:
    ```python
    haystack_pipeline.write_documents([
        {
            "title": "Claude Shannon",
            "content": "Claude Elwood Shannon was an American mathematician, "
            "electrical engineer, and cryptographer known as a father of "
            "information theory. He was a 21-year-old master's degree student at "
            "the Massachusetts Institute of Technology (MIT)."
        },
        {
            "title": "Vincent van Gogh",
            "content": "Van Gogh was born into an upper-middle-class family. "
            "As a child he was serious, quiet and thoughtful. He began drawing "
            "at an early age and as a young man worked as an art dealer."
        },
        {
            "title": "Stevie Wonder",
            "content": "Stevland Hardaway Morris, known professionally as "
            "Stevie Wonder, is an American singer and musician, who is "
            "credited as a pioneer and influence by musicians across a range "
            "of genres."
        }
    ])
    ```

    example queries:
    ```python
    from deepsparse.transformers.haystack import print_pipeline_documents
    pipeline_outputs = haystack_pipeline(
        queries="who invented information theory",
        params={"Retriever": {"top_k": 4}}
    )
    print_pipeline_documents(pipeline_outputs)

    pipeline_outputs = haystack_pipeline(
        queries=[
            "famous artists",
            "What is Stevie Wonder's real name?"
        ],
        params={"Retriever": {"top_k": 4}}
    )
    print_pipeline_documents(pipeline_outputs)
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param docs: list of documents to be written to document_store. Can also
        be written after instantiation with write_documents method.
        Default is None
    :param config: dictionary or instance of HaystackPipelineConfig. Used to
        specify Haystack node arguments
    :param retriever_kwargs: keyword arguments to be passed to retriever. If
        the retriever is a deepsparse retriever, then these arguments will also
        be passed to the EmbeddingExtractionPipeline of the retriever
    """
    return Pipeline.create("information_retrieval_haystack", *args, **kwargs)


def embedding_extraction_pipeline(*args, **kwargs) -> "Pipeline":
    """
    embedding extraction pipeline for extracting intermediate layer embeddings
    from transformer models

    example instantiation:
    ```python
    embedding_extraction_pipeline = Pipeline.create(
        task="embedding_extraction",
        model_path="masked_language_modeling_model_dir/",
    )
    results = embedding_extraction_pipeline(
        [
            "the warriors have won the nba finals"
            "the warriors are the greatest basketball team ever"
        ]
    )
    emb_1, emb_2 = results.embeddings
    # (expect emb_1 and emb_2 to have high cosine similiarity)
    ```

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
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
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param emb_extraction_layer: if an int, the transformer layer number from
        which the embeddings will be extracted. If a string, the name of last
        ONNX node in model to draw embeddings from. If None, leave the model
        unchanged. Default is -1 (last transformer layer before prediction head)
    :param model_size: size of transformer model (size of hidden layer per token
        if the model is cut). Default is 768
    :param extraction_strategy: method of pooling embedding values. Currently
        supported values are 'per_token', 'reduce_mean', 'reduce_max' and 'cls_token'.
        Default is 'per_token'
    :param return_numpy: return embeddings a list of numpy arrays, list of lists
        of floats otherwise. Default is True
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels. Default is None
    """
    return Pipeline.create("embedding_extraction", *args, **kwargs)


def zero_shot_text_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Transformers zero shot text classification pipeline. This pipeline allows for
    text classification using models which were trained on datasets not originally
    meant for this task.

    This class upon construction returns an instance of a child Pipeline which
    inherits from ZeroShotTextClassificationPipelineBase. Which type of Pipeline
    is returned depends on the value of the passed model_scheme argument.

    example dynamic labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        model_scheme="mnli",
        model_config={"hypothesis_template": "This text is related to {}"},
        model_path="mnli_model_dir/",
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    candidate_labels = ["Europe", "public health", "politics"]
    zero_shot_text_classifier(sequences=sequence_to_classify, labels=candidate_labels)
    >>> ZeroShotTextClassificationOutput(
        sequences='Who are you voting for in 2020?',
        labels=['politics', 'public health', 'Europe'],
        scores=[0.9073666334152222, 0.046810582280159, 0.04582275450229645])
    ```

    example static labels:
    ```python
    zero_shot_text_classifier = Pipeline.create(
        task="zero_shot_text_classification",
        model_scheme="mnli",
        model_config={"hypothesis_template": "This text is related to {}"},
        model_path="mnli_model_dir/",
        labels=["politics", "Europe", "public health"]
    )

    sequence_to_classify = "Who are you voting for in 2020?"
    zero_shot_text_classifier(sequences=sequence_to_classify)
    >>> ZeroShotTextClassificationOutput(
        sequences='Who are you voting for in 2020?',
        labels=['politics', 'public health', 'Europe'],
        scores=[0.9073666334152222, 0.046810582280159, 0.04582275450229645])
    ```

    Note that labels must either be provided during pipeline instantiation via
    the constructor, at inference time, but not both.

    Note that if a hypothesis_template is provided at inference time, then it
    will override the value provided during model instantiation

    :param model_path: sparsezoo stub to a transformers model or (preferred) a
        directory containing a model.onnx, tokenizer config, and model config
    :param engine_type: inference engine to use. Currently supported values include
        'deepsparse' and 'onnxruntime'. Default is 'deepsparse'
    :param batch_size: batch size must divide sequences * labels, regardless of
        whether using dynamic or static labels. Default is 1
    :param num_cores: number of CPU cores to allocate for inference engine. None
        specifies all available cores. Default is None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
    :param alias: optional name to give this pipeline instance, useful when
        inferencing with multiple models. Default is None
    :param sequence_length: sequence length to compile model and tokenizer for.
        If a list of lengths is provided, then for each length, a model and
        tokenizer will be compiled capable of handling that sequence length
        (also known as a bucket). Default is 128
    :param default_model_name: huggingface transformers model name to use to
        load a tokenizer and model config when none are provided in the `model_path`.
        Default is "bert-base-uncased"
    :param model_scheme: training scheme used to train the model used for zero shot.
        Default is "mnli"
    :param model_config: config object specific to the model_scheme of this model
        or a dict of config keyword arguments
    :param labels: static list of labels to perform text classification with. Can
        also be provided at inference time
    :param context: context for engine. If None, then the engine will be initialized
        with 2 streams to make use of parallel inference of labels
    """
    return Pipeline.create("zero_shot_text_classification", *args, **kwargs)
