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
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Context, Engine, MultiModelEngine, Scheduler
from deepsparse.base_pipeline import _REGISTERED_PIPELINES, BasePipeline, SupportedTasks
from deepsparse.benchmark import ORTEngine, TorchScriptEngine
from deepsparse.cpu import cpu_details
from deepsparse.loggers.base_logger import BaseLogger
from deepsparse.loggers.constants import MetricCategories, SystemGroups
from deepsparse.utils import (
    InferenceStages,
    StagedTimer,
    TimerManager,
    join_engine_outputs,
    split_engine_inputs,
)


__all__ = [
    "DEEPSPARSE_ENGINE",
    "ORT_ENGINE",
    "TORCHSCRIPT_ENGINE",
    "SUPPORTED_PIPELINE_ENGINES",
    "Pipeline",
    "BasePipeline",
    "SupportedTasks",
    "_REGISTERED_PIPELINES",
    "PipelineConfig",
    "question_answering_pipeline",
    "text_classification_pipeline",
    "zero_shot_text_classification_pipeline",
    "token_classification_pipeline",
    "image_classification_pipeline",
    "yolo_pipeline",
    "Bucketable",
    "BucketingPipeline",
    "create_engine",
    "TextGeneration",
    "CodeGeneration",
    "Chat",
]

DEEPSPARSE_ENGINE = "deepsparse"
ORT_ENGINE = "onnxruntime"
TORCHSCRIPT_ENGINE = "torchscript"

SUPPORTED_PIPELINE_ENGINES = [DEEPSPARSE_ENGINE, ORT_ENGINE]


class Pipeline(BasePipeline):
    """
    Generic Pipeline abstract class meant to wrap inference engine objects to include
    data pre/post-processing. Inputs and outputs of pipelines should be serialized
    as pydantic Models. See the BasePipeline above for additional parameters provided
    during inference.

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
    :param num_streams: The max number of requests the model can handle
        concurrently. None or 0 implies a scheduler-defined default value;
        default None
    :param scheduler: (deepsparse only) kind of scheduler to execute with.
        Pass None for the default
    :param input_shapes: list of shapes to set ONNX the inputs to. Pass None
        to use model as-is. Default is None
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
        num_streams: int = None,
        scheduler: Scheduler = None,
        input_shapes: List[List[int]] = None,
        context: Optional[Context] = None,
        executor: Optional[Union[ThreadPoolExecutor, int]] = None,
        benchmark: bool = False,
        _delay_engine_initialize: bool = False,  # internal use only
        **kwargs,
    ):
        self._benchmark = benchmark
        self._model_path_orig = model_path
        self._model_path = model_path
        self._engine_type = engine_type
        self._batch_size = batch_size
        self._timer_manager = TimerManager(enabled=True, multi=benchmark)
        self.context = context
        super().__init__(**kwargs)

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
            self._engine_args["num_streams"] = num_streams

        self.onnx_file_path = self.setup_onnx_file_path()

        if _delay_engine_initialize:
            self.engine = None
        else:
            self.engine = self._initialize_engine()
        self._batch_size = self._batch_size or 1

        self.log(
            identifier=f"{SystemGroups.INFERENCE_DETAILS}/num_cores_total",
            value=num_cores,
            category=MetricCategories.SYSTEM,
        )

    def __call__(self, *args, **kwargs) -> BaseModel:
        with self.timer_manager.new_timer_context() as timer:
            if "engine_inputs" in kwargs:
                raise ValueError(
                    "invalid kwarg engine_inputs. engine inputs determined "
                    f"by {self.__class__.__qualname__}.parse_inputs"
                )

            # ------ PREPROCESSING ------
            timer.start(InferenceStages.PRE_PROCESS)
            # parse inputs into input_schema
            pipeline_inputs = self.parse_inputs(*args, **kwargs)
            self.log(
                identifier="pipeline_inputs",
                value=pipeline_inputs,
                category=MetricCategories.DATA,
            )

            if not isinstance(pipeline_inputs, self.input_schema):
                raise RuntimeError(
                    f"Unable to parse {self.__class__} inputs into a "
                    f"{self.input_schema} object. "
                    f"Inputs parsed to {type(pipeline_inputs)}"
                )
            # batch size of the inputs may be `> self._batch_size` at this point
            engine_inputs = self.process_inputs(pipeline_inputs)
            if isinstance(engine_inputs, tuple):
                engine_inputs, context = engine_inputs
            else:
                context = {}

            timer.stop(InferenceStages.PRE_PROCESS)
            self.log(
                identifier="engine_inputs",
                value=engine_inputs,
                category=MetricCategories.DATA,
            )

            # ------ INFERENCE ------
            # split inputs into batches of size `self._batch_size`
            timer.start(InferenceStages.ENGINE_FORWARD)
            batches, orig_batch_size = self.split_engine_inputs(
                engine_inputs, self._batch_size
            )

            # submit split batches to engine threadpool
            engine_forward_with_context = partial(self.engine_forward, context=context)
            batch_outputs = list(
                self.executor.map(engine_forward_with_context, batches)
            )

            # join together the batches of size `self._batch_size`
            engine_outputs = self.join_engine_outputs(
                batch_outputs, orig_batch_size, **context
            )
            timer.stop(InferenceStages.ENGINE_FORWARD)

            self.log(
                identifier=f"{SystemGroups.INFERENCE_DETAILS}/input_batch_size_total",
                # to get the batch size of the inputs, we need to look
                # to multiply the engine batch size (self._batch_size)
                # by the number of batches processed by the engine during
                # a single inference call
                value=len(batch_outputs) * self._batch_size,
                category=MetricCategories.SYSTEM,
            )
            self.log(
                identifier="engine_outputs",
                value=engine_outputs,
                category=MetricCategories.DATA,
            )

            # ------ POSTPROCESSING ------
            timer.start(InferenceStages.POST_PROCESS)
            pipeline_outputs = self.process_engine_outputs(engine_outputs, **context)
            if not isinstance(pipeline_outputs, (self.output_schema, Generator)):
                raise ValueError(
                    f"Outputs of {self.__class__} must be instances of "
                    f"{self.output_schema} found output of type "
                    f"{type(pipeline_outputs)}"
                )
            timer.stop(InferenceStages.POST_PROCESS)
            self.log(
                identifier="pipeline_outputs",
                value=pipeline_outputs,
                category=MetricCategories.DATA,
            )

        self.log_inference_times(timer)

        return pipeline_outputs

    @classmethod
    def from_config(
        cls,
        config: Union["PipelineConfig", str, Path],
        context: Optional[Context] = None,
        logger: Optional[BaseLogger] = None,
    ) -> "Pipeline":
        """
        :param config: PipelineConfig object, filepath to a json serialized
            PipelineConfig, or raw string of a json serialized PipelineConfig
        :param context: Optional Context object to use for creating instances of
            MultiModelEngine. The Context contains a shared scheduler along with
            other runtime information that will be used across instances of the
            MultiModelEngine to provide optimal performance when running
            multiple models concurrently
        :param logger: An optional DeepSparse Logger object for inference
            logging. Default is None
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
            logger=logger,
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

    @property
    def timer_manager(self) -> TimerManager:
        return self._timer_manager

    @property
    def current_timer(self) -> Optional[StagedTimer]:
        """
        :return: current timer for the pipeline, if any
        """
        timer = self.timer_manager.current

        if timer is None:
            timer = self.timer_manager.latest

        return timer

    @property
    def benchmark(self) -> bool:
        return self._benchmark

    @benchmark.setter
    def benchmark(self, value: bool):
        self._benchmark = value
        self.timer_manager.multi = value

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
            batch_size=self._batch_size,
            num_cores=self._engine_args.get("num_cores"),
            scheduler=self._engine_args.get("scheduler"),
            input_shapes=self._engine_args.get("input_shapes"),
            alias=self.alias,
            kwargs=kwargs,
        )

    def join_engine_outputs(
        self, batch_outputs: List[List[numpy.ndarray]], orig_batch_size: int, **kwargs
    ) -> List[numpy.ndarray]:
        """
        Joins list of engine outputs together into one list.
        This is the opposite of `split_engine_inputs` and is meant to be used in tandem.

        :param batch_outputs: list of engine outputs
        :param orig_batch_size: original batch size of the inputs
        :return: list of engine outputs joined together
        """
        return join_engine_outputs(batch_outputs, orig_batch_size)

    def split_engine_inputs(
        self, items: List[numpy.ndarray], batch_size: int
    ) -> List[List[numpy.ndarray]]:
        """
        Splits each item into numpy arrays with the first dimension == `batch_size`.
        This is the opposite of `join_engine_outputs` and is meant to be used in tandem.

        :param items: size of each batch to split into
        :param batch_size: size of each batch to enforce

        :return: list of batches, where each batch is a list of numpy arrays
        """
        return split_engine_inputs(items, batch_size)

    def engine_forward(
        self,
        engine_inputs: List[numpy.ndarray],
        context: Dict = {},
    ) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :param context: optional dictionary to be used during engine execution
        :return: result of forward pass to Pipeline engine
        """
        return self.engine(engine_inputs)

    def log_inference_times(self, timer: StagedTimer):
        """
        logs stage times in the given timer

        :param timer: timer to log
        """
        for stage, time in timer.times.items():
            self.log(
                identifier=f"{SystemGroups.PREDICTION_LATENCY}/{stage}_seconds",
                value=time,
                category=MetricCategories.SYSTEM,
            )

    def _initialize_engine(
        self,
    ) -> Union[Engine, MultiModelEngine, ORTEngine, TorchScriptEngine]:
        return create_engine(
            self.onnx_file_path, self.engine_type, self._engine_args, self.context
        )

    def _properties_dict(self) -> Dict:
        return {
            "config": self.to_config(),
            "engine": self.engine,
        }

    def __repr__(self):
        """
        :return: Unambiguous representation of the current pipeline
        """
        return "{}({})".format(self.__class__, self._properties_dict())

    def __str__(self):
        """
        :return: Human readable form of the current pipeline
        """
        formatted_props = [
            "\t{}: {}".format(key, val) for key, val in self._properties_dict().items()
        ]

        return "{}.{}:\n{}".format(
            self.__class__.__module__,
            self.__class__.__qualname__,
            "\n".join(formatted_props),
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
        default=None,
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
    scheduler: Optional[str] = Field(
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


def create_engine(
    onnx_file_path: str,
    engine_type: str,
    engine_args: Dict,
    context: Optional[Context] = None,
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
        if context is not None and isinstance(context, Context):
            engine_args.pop("num_cores", None)
            engine_args.pop("scheduler", None)
            engine_args.pop("num_streams", None)
            engine_args["context"] = context
            return MultiModelEngine(
                model=onnx_file_path,
                **engine_args,
            )
        engine_args.pop("cache_output_bools", None)
        return Engine(onnx_file_path, **engine_args)

    if engine_type == ORT_ENGINE:
        return ORTEngine(onnx_file_path, **engine_args)

    if engine_type == TORCHSCRIPT_ENGINE:
        return TorchScriptEngine(onnx_file_path, **engine_args)

    raise ValueError(
        f"Unknown engine_type {engine_type}. Supported values include: "
        f"{SUPPORTED_PIPELINE_ENGINES}"
    )


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


def text_generation_pipeline(
    *args, model: Optional[str] = None, **kwargs
) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _parse_model_arg(model, **kwargs)
    return Pipeline.create("text_generation", *args, **kwargs)


def code_generation_pipeline(
    *args, model: Optional[str] = None, **kwargs
) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _parse_model_arg(model, **kwargs)
    return Pipeline.create("code_generation", *args, **kwargs)


def chat_pipeline(*args, model: Optional[str] = None, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _parse_model_arg(model, **kwargs)
    return Pipeline.create("chat", *args, **kwargs)


def _parse_model_arg(model: Optional[str], **kwargs) -> dict:
    if model is not None:
        model_path = kwargs.get("model_path")
        if model_path is not None:
            raise ValueError(
                f"Only one of model and model_path may be supplied, found {model} "
                f"and {model_path} respectively"
            )
        kwargs["model_path"] = model
    return kwargs


# aliases for top level import
TextGeneration = text_generation_pipeline
CodeGeneration = code_generation_pipeline
Chat = chat_pipeline


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
        be passed to the TransformersEmbeddingExtractionPipeline of the retriever
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
