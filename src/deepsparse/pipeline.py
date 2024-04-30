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
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Union

from deepsparse.loggers.logger_manager import LoggerManager
from deepsparse.middlewares import IS_NESTED_KEY, NAME_KEY, MiddlewareManager
from deepsparse.operators import EngineOperator, Operator
from deepsparse.pipeline_config import PipelineConfig
from deepsparse.routers import Router
from deepsparse.schedulers import (
    ContinuousBatchingScheduler,
    OperatorScheduler,
    SchedulerGroup,
)
from deepsparse.subgraph_execute import SubGraphExecutor
from deepsparse.tasks import SupportedTasks
from deepsparse.utils import InferenceState, PipelineState
from deepsparse.utils.subgraph import SubGraph
from deepsparse.utils.time import TIMER_KEY, InferenceStages, TimerManager


__all__ = [
    "Pipeline",
    "TextGeneration",
    "CodeGeneration",
    "Chat",
    "question_answering_pipeline",
    "text_classification_pipeline",
    "zero_shot_text_classification_pipeline",
    "token_classification_pipeline",
    "haystack_pipeline",
    "sentiment_analysis_pipeline",
    "embedding_extraction_pipeline",
    "image_classification_pipeline",
    "yolo_pipeline",
]

_LOGGER = logging.getLogger(__name__)
V2_NOT_SUPPORTED = ["alias", "logger", "executor"]


class Pipeline(Operator):
    """
    Pipeline accepts a series of operators, schedulers, and a router. Calling a pipeline
    will use the router to run through all the defined operators. The operators should
    be implemented using the Operator class and each implemented operator should be
    responsible for a functional component of the pipelines. The flow of inputs/outputs
    between the operators and the steps in the pipeline should be defined by the router,
    (based off of the Router class), which dicates the next operator in the pipeline.
    Execution of the operators will be handled by the provided schedulers.

    :param ops: Operators to run within the pipeline. Can either be a list of operators
        or dictionary of operators.
    :param router: A Router which dictates the next operator to call.
    :param schedulers: A list of schedulers to run operators.
    :param pipeline_state: pipeline_state created during pipeline initialization
    :param middleware_manager: middlewares to be used in Pipeline and Operator
    :param timer_manager: instantiated TimerManger to track timings

    """

    def __init__(
        self,
        ops: Union[Dict[str, Operator], List[Operator]],
        router: Router,
        schedulers: List[OperatorScheduler],
        generator_router: Optional[Router] = None,
        continuous_batching_scheduler: Optional[ContinuousBatchingScheduler] = None,
        pipeline_state: Optional[PipelineState] = None,
        middleware_manager: Optional[MiddlewareManager] = None,
        timer_manager: Optional[TimerManager] = None,
        logger_manager: Optional[LoggerManager] = None,
        benchmark: bool = False,
    ):

        self.ops = ops
        self.router = router
        self.generator_router = generator_router
        self.schedulers = schedulers
        self.pipeline_state = pipeline_state
        self._continuous_batching_scheduler = continuous_batching_scheduler
        self.middleware_manager = middleware_manager
        self.timer_manager = timer_manager or TimerManager()
        self.logger_manager = logger_manager or LoggerManager()
        self.validate()

        self._scheduler_group = SchedulerGroup(self.schedulers)
        self.subgraph_executor = SubGraphExecutor()
        self.benchmark = benchmark

    @property
    def input_schema(self):
        raise AttributeError("No input schema has been set for this pipeline.")

    @property
    def output_schema(self):
        raise AttributeError("No output schema has been set for this pipeline.")

    @classmethod
    def create(cls, task: str, **kwargs) -> "Pipeline":
        """
        :param task: Pipeline task
        :param kwargs: extra task specific kwargs to be passed to the Pipeline
        :return: pipeline object initialized for the given task
        """
        new_kwargs = {}
        for k in kwargs:
            if k in V2_NOT_SUPPORTED:
                _LOGGER.warning(f"{k} is not yet supported in the v2 pipeline.")
            else:
                new_kwargs[k] = kwargs.get(k)

        try:
            model_path = new_kwargs.get("model_path")
            model = new_kwargs.pop("model", None)

            if model and model_path:
                raise ValueError(
                    f"Only one of model and model_path may be supplied, found {model} "
                    f"and {model_path} respectively"
                )
            elif model:
                new_kwargs["model_path"] = model

            pipeline = Operator.create(task=task, **new_kwargs)
            if not isinstance(pipeline, cls):
                raise RuntimeError(
                    "Pipeline was not created for the given task. The "
                    "provided task should be registered using the OperatorRegistry"
                )
        except Exception as e:
            if SupportedTasks.is_text_generation(task):
                raise e

            _LOGGER.warning(f"Could not create v2 '{task}' pipeline, trying legacy")
            from deepsparse.legacy import Pipeline

            pipeline = Pipeline.create(task=task, **kwargs)
        return pipeline

    @classmethod
    def from_config(
        cls, config: Union["PipelineConfig", str, Path], **kwargs
    ) -> "Pipeline":
        """
        :param config: PipelineConfig object, filepath to a json serialized
            PipelineConfig, or raw string of a json serialized PipelineConfig.
            Optionally, pipeline arguments not defined in the PipelineConfig may be
            passed as key-word arguments to this function.
        """
        if isinstance(config, Path) or (
            isinstance(config, str) and os.path.exists(config)
        ):
            if isinstance(config, str):
                config = Path(config)
            config = PipelineConfig.parse_file(config)
        if isinstance(config, str):
            config = PipelineConfig.parse_raw(config)

        kwargs.update(config.kwargs)
        return cls.create(
            task=config.task,
            model_path=config.model_path,
            engine_type=config.engine_type,
            batch_size=config.batch_size,
            num_cores=config.num_cores,
            scheduler=config.scheduler,
            input_shapes=config.input_shapes,
            alias=config.alias,
            **kwargs,
        )

    async def run_async(self, *args, inference_state: InferenceState, **kwargs):
        """
        Run through the operators using the provided router and scheduler.
        The input to a given operator is the output of the previous operator.

        :param inference_state: inference_state for the pipeline.
        """
        loop = asyncio.get_running_loop()

        next_step = self.router.START_ROUTE
        operator_output = None
        if (
            not hasattr(inference_state, TIMER_KEY)
            or getattr(inference_state, TIMER_KEY) is None
        ):
            timer = self.timer_manager.get_new_timer()
            inference_state.set_timer(timer)
        if (
            not hasattr(inference_state, "logger")
            or getattr(inference_state, "logger") is None
        ):
            inference_state.set_logger(self.logger_manager.metric)

        with inference_state.time(id=InferenceStages.TOTAL_INFERENCE):
            while next_step != self.router.END_ROUTE:
                # Check if running streaming; if that is the case, will return
                # an AsyncGenerator. This requires the pipeline to support
                # streaming with a generator_router set
                if inference_state.current_state.get("streaming"):
                    return self._run_generate_async(
                        operator_output=operator_output,
                        inference_state=inference_state,
                        next_step=next_step,
                        loop=loop,
                    )

                # Non Streaming/Generator pathway
                if next_step == self.router.SPLIT_ROUTE:
                    if operator_output is None:
                        raise ValueError(
                            f"{self.router.SPLIT_ROUTE} should appear after "
                            f"{self.ROUTER.START_ROUTE}"
                        )

                    operator_output = await self._apply_split_async(
                        operator_output, inference_state, loop=loop
                    )
                    next_step = self.router.JOIN_ROUTE

                else:
                    if next_step == self.router.START_ROUTE:
                        outputs = self.run_func(
                            *args,
                            func=self._scheduler_group.submit,
                            operator=self.ops[next_step],
                            inference_state=inference_state,
                            pipeline_state=self.pipeline_state,
                            loop=loop,
                            **kwargs,
                        )
                    else:
                        outputs = self._run_next(
                            inp=operator_output,
                            next_step=next_step,
                            inference_state=inference_state,
                            loop=loop,
                        )

                    await outputs
                    operator_output = outputs.result()

                    if isinstance(operator_output, tuple):
                        operator_output, state_update = (
                            operator_output[0],
                            operator_output[-1],
                        )
                        inference_state.update_state(state_update)

                next_step = self.router.next(next_step, self.ops, operator_output)

            rtn = operator_output

        self.timer_manager.update(inference_state.timer.measurements)
        return rtn

    def run(
        self,
        *args,
        inference_state: InferenceState,
        **kwargs,
    ):
        """
        Run through the operators using the provided router and scheduler.
        The input to a given operator is the output of the previous operator.

        :param inference_state: inference_state for the pipeline.
        """
        next_step = self.router.START_ROUTE
        operator_output = None
        while next_step != self.router.END_ROUTE:
            # Check if running streaming; if that is the case, will return
            # a Generator. This requires the pipeline to support
            # streaming with a generator_router set.
            if inference_state.current_state.get("streaming"):
                return self._run_generate(
                    operator_output=operator_output,
                    inference_state=inference_state,
                    next_step=next_step,
                )

            # Non Streaming/Generator pathway
            if next_step == self.router.SPLIT_ROUTE:
                if operator_output is None:
                    raise ValueError(
                        f"{self.router.SPLIT_ROUTE} should appear after "
                        f"{self.router.START_ROUTE}"
                    )
                operator_output = self._apply_split(operator_output, inference_state)
                next_step = self.router.JOIN_ROUTE

            else:
                if next_step == self.router.START_ROUTE:
                    operator_output = self.run_func(
                        *args,
                        func=self._scheduler_group.submit,
                        operator=self.ops[next_step],
                        inference_state=inference_state,
                        pipeline_state=self.pipeline_state,
                        **kwargs,
                    ).result()
                else:
                    operator_output = self._run_next(
                        inp=operator_output,
                        next_step=next_step,
                        inference_state=inference_state,
                    ).result()

                if isinstance(operator_output, tuple):
                    operator_output, state_update = (
                        operator_output[0],
                        operator_output[-1],
                    )
                    inference_state.update_state(state_update)

            next_step = self.router.next(next_step, self.ops, operator_output)
        return operator_output

    def _run_generate(
        self,
        operator_output: Any,
        inference_state: InferenceState,
        next_step: str,
    ) -> Generator:

        """
        Run pipeline execution in streaming/generator mode. _run_generate will run
        the same loop with stop conditions as run() but will return a Generator.

        :param operator_output: previous operator output, used as input for the next
            operator.
        :param inference_state: inference_state for the pipeline.
        :param next_step: string indicating the next step to run
        """
        if not self.generator_router:
            raise ValueError("For streaming mode, a generator_router must be provided.")

        while next_step != self.generator_router.END_ROUTE:
            start_step = next_step

            if next_step == self.router.SPLIT_ROUTE:
                end = [self.generator_router.JOIN_ROUTE]
                step = self.generator_router.route[self.generator_router.SPLIT_ROUTE]
                initial_inference_state = inference_state
            else:
                step = next_step
                end = [
                    self.generator_router.SPLIT_ROUTE,
                    self.generator_router.END_ROUTE,
                ]

            for output in self._apply_split_generation(
                operator_output, inference_state, step, end
            ):
                output_to_yield, next_step, operator_output, inference_state = output
                yield output_to_yield

            if start_step == self.generator_router.SPLIT_ROUTE:
                inference_state = initial_inference_state

            next_step = self.generator_router.next(next_step, self.ops, operator_output)

    async def _run_generate_async(
        self,
        operator_output: Any,
        inference_state: InferenceState,
        next_step: str,
        loop: asyncio.AbstractEventLoop,
    ) -> AsyncGenerator:

        """
        Run pipeline execution in streaming/generator mode. _run_generate_async will run
        the same loop with stop conditions as run_async() but will return an
        AsyncGenerator.

        :param operator_output: previous operator output, used as input for the next
            operator.
        :param inference_state: inference_state for the pipeline.
        :param next_step: string indicating the next step to run
        """
        if not self.generator_router:
            raise ValueError("For streaming mode, a generator_router must be provided.")

        while next_step != self.generator_router.END_ROUTE:
            start_step = next_step

            if next_step == self.router.SPLIT_ROUTE:
                end = [self.generator_router.JOIN_ROUTE]
                step = self.generator_router.route[self.generator_router.SPLIT_ROUTE]
                initial_inference_state = inference_state
            else:
                step = next_step
                end = [
                    self.generator_router.SPLIT_ROUTE,
                    self.generator_router.END_ROUTE,
                ]

            async for output in self._apply_split_generation_async(
                operator_output, inference_state, step, end, loop
            ):
                output_to_yield, next_step, operator_output, inference_state = output
                yield output_to_yield

            if start_step == self.generator_router.SPLIT_ROUTE:
                inference_state = initial_inference_state

            # TODO: might need additional processing on operator_output with more
            # complex grapghs
            next_step = self.generator_router.next(next_step, self.ops, operator_output)

    def __call__(self, *args, **kwargs):
        """
        Consolidate any provided inference_state or pipeline_state objects and pass
        any other operator inputs to run().

        :return: output of the pipeline operators ran with the router for the given
            input
        """
        is_nested = True
        if kwargs.get("inference_state"):
            inference_state = kwargs.pop("inference_state")
        else:
            inference_state = InferenceState()
            inference_state.create_state({})

            timer = self.timer_manager.get_new_timer()
            inference_state.set_timer(timer)

            inference_state.set_logger(self.logger_manager.metric)

            is_nested = False

        kwargs["inference_state"] = inference_state
        kwargs[NAME_KEY] = InferenceStages.TOTAL_INFERENCE
        kwargs[IS_NESTED_KEY] = is_nested

        # timer shared across all operators, has all measurements
        timer = inference_state.timer

        next_call = self.run
        if self.middleware_manager is not None:
            # make next calls to be middlewares if any
            next_call = self.middleware_manager.build_middleware_stack(next_call)

        rtn = next_call(*args, **kwargs)

        # update all the measurments
        self.timer_manager.update(timer.measurements)

        return rtn

    def expand_inputs(self, *args, **kwargs):
        """
        Generic function to handle expanding values.
        """
        raise NotImplementedError(
            "This function should be implemented for any router with split or join"
            "nodes. expand_inputs will be called prior to the split node (stored in "
            "the router's SPLIT_ROUTE attribute), expanding outputs for each output "
            "such that there is a batch size of one per thread."
        )

    def condense_inputs(self, *args, **kwargs):
        """
        Generic function to handle condensing values.
        """
        raise NotImplementedError(
            "This function should be implemented for any router with split or join "
            "nodes. condense_inputs will be called after the join node (stored in the "
            "router's JOIN_ROUTE attribute), condensing outputs from multiple threads."
        )

    def validate(self):
        """
        Validate that compatability of the router and operators provided.
        """
        router_validation = self.router.validate(self.ops)

        if router_validation is False:
            # default error message
            op_types = [type(op) for op in self.ops]
            raise ValueError(f"Invalid Router: {type(self.router)} for ops: {op_types}")
        elif isinstance(router_validation, str):
            raise ValueError(f"Invalid Router for operators: {router_validation}")

        if (
            self.middleware_manager is not None
            and self._continuous_batching_scheduler is not None
        ):
            _LOGGER.warning(
                "Middleware is yet to be supported using continous batching scheduler. "
                "Either remove middleware or remove continous batching scheduler "
                "in the instantiation of the Pipeline class"
            )

    def run_func(
        self,
        *args,
        operator: Operator,
        func: Callable,
        inp: Any = None,
        **kwargs,
    ):
        """
        Wrap the operator with middleware and execute the func callable.
        InferenceState, PipelineState is inside kwargs

        :param operator: Operator instance
        :param func: Desired function to call. Ex. SchedulerGroup.submit
        :param inp: Any input to the operator. Ex. IntSchema
        """

        # wrap the operator with the middleware, if any
        wrapped_operator = operator
        if self.middleware_manager is not None:
            wrapped_operator = self.middleware_manager.wrap(operator)

        kwargs["operator"] = wrapped_operator

        if isinstance(inp, dict):
            if NAME_KEY not in inp:
                kwargs[NAME_KEY] = operator.__class__.__name__
        else:
            kwargs[NAME_KEY] = operator.__class__.__name__

        if inp:
            output = (
                func(*args, **kwargs, **inp)
                if isinstance(inp, dict)
                else func(inp, *args, **kwargs)
            )
        else:
            output = func(*args, **kwargs)

        return output

    def _apply_split(self, inp: Any, inference_state: InferenceState):
        """
        Split the data provided into batch sizes of 1. Create subgraphs with each batch
        and execute the subgraph. Condense the outputs together when all subgraphs have
        finished running and return.

        :param inp: input to the operators
        :param inference_state: InferenceState for the operators
        """
        batches, orig_batch_size = self.expand_inputs(inp, 1)

        step = self.router.route[self.router.SPLIT_ROUTE]
        end = [self.router.JOIN_ROUTE]
        split_graphs = self._create_and_start_subgraph(
            inference_state=inference_state, data=batches, step=step, end=end
        )
        outputs = self.subgraph_executor.run_sub_graphs(
            router=self.router,
            ops=self.ops,
            func=self._run_next,
            sub_graphs=split_graphs,
        )
        return self.condense_inputs(outputs)

    async def _apply_split_async(
        self, inp: Any, inference_state: InferenceState, loop: asyncio.AbstractEventLoop
    ):
        """
        Split the data provided into batch sizes of 1. Create subgraphs with each batch
        and execute the subgraph. Condense the outputs together when all subgraphs have
        finished running and return.

        :param inp: input to the operators
        :param inference_state: InferenceState for the operators
        """
        batches, orig_batch_size = self.expand_inputs(inp, 1)

        step = self.router.route[self.router.SPLIT_ROUTE]
        end = [self.router.JOIN_ROUTE]
        split_graphs = self._create_and_start_subgraph(
            inference_state=inference_state, data=batches, step=step, end=end, loop=loop
        )
        outputs = await self.subgraph_executor.run_sub_graphs_async(
            router=self.router,
            ops=self.ops,
            func=self._run_next,
            sub_graphs=split_graphs,
            loop=loop,
        )
        return self.condense_inputs(outputs)

    async def _apply_split_generation_async(
        self,
        inp: Any,
        inference_state: InferenceState,
        step: str,
        end: List[str],
        loop: asyncio.AbstractEventLoop,
    ) -> AsyncGenerator:
        """
        Applies the same logic as _apply_split_async but returns an AsycnGenerator.
        """
        batches, orig_batch_size = self.expand_inputs(inp, 1)

        for i in range(len(batches)):
            split_graphs = self._create_and_start_subgraph(
                inference_state=inference_state,
                data=[batches[i]],
                step=step,
                end=end,
                loop=loop,
            )
            async for output in self.subgraph_executor.run_sub_graphs_async_generator(
                router=self.generator_router,
                ops=self.ops,
                func=self._run_next,
                sub_graphs=split_graphs,
                loop=loop,
            ):
                yield output

    def _apply_split_generation(
        self, inp: Any, inference_state: InferenceState, step: str, end: List[str]
    ) -> Generator:
        """
        Applies the same logic as _apply_split but returns a Generator.
        """
        batches, orig_batch_size = self.expand_inputs(inp, 1)

        for i in range(len(batches)):
            split_graphs = self._create_and_start_subgraph(
                inference_state=inference_state, data=[batches[i]], step=step, end=end
            )
            for output in self.subgraph_executor.run_sub_graphs_generator(
                router=self.generator_router,
                ops=self.ops,
                func=self._run_next,
                sub_graphs=split_graphs,
            ):
                yield output

    def _create_and_start_subgraph(
        self,
        inference_state: InferenceState,
        data: List[Any],
        step: str,
        end: List[str],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[SubGraph]:
        """
        Create SubGraphs given a list of data and an InferenceState objects. A SubGraph
        will be created for each each item in the data list and a copy of the
        InferenceState. Once created, the the first Operator of the SubGraph will be
        scheduled and a list of the SubGraphs will be returned.

        :param inference_state: InferenceState Object
        :param data: list of data to execute the operators with
        :param step: the starting operator step
        :parm end: list of steps indicating when the SubGraph has finished running
        """
        graphs = [
            SubGraph(
                inf=inference_state.copy_state(),
                step=step,
                end=end,
            )
            for i in range(len(data))
        ]
        split_graphs = self.subgraph_executor.start_subgraphs(
            func=self._run_next, sub_graph_inputs=data, sub_graphs=graphs, loop=loop
        )
        return split_graphs

    def _run_next(
        self, inp: Any, inference_state: InferenceState, next_step: str, **kwargs
    ):
        """
        Function to schedule the operator. If a continuous_batching_scheduler is
        provided, all operators deriving from the EngineOperator will be scheduled
        using this scheduler. All other operators will be scheduled using the
        default scheduler.

        :param inp: input to the operator
        :param inference_state: inference state for the operator
        :param next_step: dictionary key to fetch the operator from the pipeline ops.
        """
        if (
            isinstance(self.ops[next_step], EngineOperator)
            and self._continuous_batching_scheduler
        ):
            func = self._continuous_batching_scheduler.submit
            inp = self.ops[next_step].input_schema(**inp)
        else:
            func = self._scheduler_group.submit

        return self.run_func(
            func=func,
            operator=self.ops[next_step],
            inp=inp,
            pipeline_state=self.pipeline_state,
            inference_state=inference_state,
            **kwargs,
        )


def text_generation_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _check_model_path_arg(*args, **kwargs)
    return Pipeline.create("text_generation", **kwargs)


def code_generation_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _check_model_path_arg(*args, **kwargs)
    return Pipeline.create("code_generation", **kwargs)


def chat_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    kwargs = _check_model_path_arg(*args, **kwargs)
    return Pipeline.create("chat", **kwargs)


TextGeneration = text_generation_pipeline
CodeGeneration = code_generation_pipeline
Chat = chat_pipeline


def question_answering_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers question_answering pipeline
    """

    return Pipeline.create("question_answering", *args, **kwargs)


def text_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers text classification pipeline
    """

    return Pipeline.create("text_classification", *args, **kwargs)


def sentiment_analysis_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers text classification pipeline

    """
    return Pipeline.create("text_classification", *args, **kwargs)


def token_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    transformers token classification pipeline
    """

    return Pipeline.create("token_classification", *args, **kwargs)


def image_classification_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Image classification pipeline for DeepSparse
    """

    return Pipeline.create("image_classification", *args, **kwargs)


def yolo_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Image Segmentation YOLO pipeline for DeepSparse

    """
    return Pipeline.create("yolo", *args, **kwargs)


def haystack_pipeline(*args, **kwargs) -> "Pipeline":
    """
    Neural Magic pipeline for running Haystack DocumentSearchPipeline.
    Supports selected Haystack Nodes as well as Haystack nodes integrated
    with the Neural Magic DeepSparse Engine

    Note: Deprecated due to lack of pydanticV2 support in Haystack v1
    """
    raise DeprecationWarning(
        "Haystack support with deepsparse has been deprecated, "
        "kindly use deepsparse-nightly==1.8.20240404 or older"
    )


def embedding_extraction_pipeline(*args, **kwargs) -> "Pipeline":
    """
    embedding extraction pipeline for extracting intermediate layer embeddings
    from transformer models
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
    """
    return Pipeline.create("zero_shot_text_classification", *args, **kwargs)


def _check_model_path_arg(*args, **kwargs):
    if args:
        if len(args) > 1 or "model_path" in kwargs or "model" in kwargs:
            raise ValueError(
                "Only the model path can be provided as a non-kwarg argument"
            )
        kwargs["model_path"] = args[0]
    return kwargs
