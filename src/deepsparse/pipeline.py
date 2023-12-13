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
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deepsparse.operators import EngineOperator, Operator
from deepsparse.pipeline_config import PipelineConfig
from deepsparse.routers import Router
from deepsparse.schedulers import (
    ContinuousBatchingScheduler,
    OperatorScheduler,
    SchedulerGroup,
)
from deepsparse.utils import InferenceState, PipelineState
from deepsparse.utils.helpers import run_func
from deepsparse.utils.subgraph import SubGraph


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

    """

    def __init__(
        self,
        ops: Union[Dict[str, Operator], List[Operator]],
        router: Router,
        schedulers: List[OperatorScheduler],
        continuous_batching_scheduler: Optional[ContinuousBatchingScheduler] = None,
        pipeline_state: Optional[PipelineState] = None,
    ):

        self.ops = ops
        self.router = router
        self.schedulers = schedulers
        self.pipeline_state = pipeline_state
        self._continuous_batching_scheduler = continuous_batching_scheduler
        self.validate()

        self._scheduler_group = SchedulerGroup(self.schedulers)

    def _run_next(
        self, inp: Any, inference_state: InferenceState, next_step: str, **kwargs
    ):
        if (
            isinstance(self.ops[next_step], EngineOperator)
            and self._continuous_batching_scheduler
        ):
            func = self._continuous_batching_scheduler.submit
            inp = self.ops[next_step].input_schema(**inp)
        else:
            func = self._scheduler_group.submit

        return run_func(
            func=func,
            operator=self.ops[next_step],
            inp=inp,
            pipeline_state=self.pipeline_state,
            inference_state=inference_state,
            **kwargs,
        )

    async def _run_sub_graphs(
        self,
        sub_graph_inputs: List[Any],
        sub_graphs: List[SubGraph],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> List[Any]:
        """
        Run a list of sub_graphs asynchronously. Polls to identify the sub graph that is
        still running but has completed its current step. Schedules the next step
        subgraph step. This is repeated until all subgraphs have finished running and
        have reached their end step (stored in the Subgraph.end attribute).

        :param sub_graph_inputs: A list of inputs that should be passed to each
        subgraph. Each subgraph is given an element of the list as input to its
        first node.
        :param sub_graphs: A list of Subgraph objects. Each stores the relevant
        execution information for the particular subgraph, such as its current step
        in the sub graph, inference state, output, and end step.

        :returns: a list of outputs for all the completed Subgraph objects. Returned
        in the same order that the subgraphs were passed to the function.
        """
        for i in range(len(sub_graphs)):
            sub_graphs[i].output = self._run_next(
                sub_graph_inputs[i], sub_graphs[i].inf, sub_graphs[i].step, loop=loop
            )

        # Execute all sub graphs until all graphs have been completed.
        while any(not x.completed for x in sub_graphs):
            for sub_graph in sub_graphs:
                if not sub_graph.completed:
                    # get the result for the completed operator; resolve its output
                    if isinstance(sub_graph.output, asyncio.Future):
                        await sub_graph.output
                    operator_output = sub_graph.output.result()
                    operator_output = sub_graph.parse_output(operator_output)

                    # determine the next step for the particular operator, using
                    # its previous output and previously stored step
                    next_step = self.router.next(
                        sub_graph.step, self.ops, operator_output
                    )
                    # update the step
                    sub_graph.step = next_step

                    # store the output for the next step. If the next step is
                    # end step, this particular route has completed. Simply
                    # update the output value
                    if next_step in sub_graph.end:
                        sub_graph.output = operator_output
                        sub_graph.completed = True
                    else:
                        sub_graph.output = self._run_next(
                            inp=operator_output,
                            inference_state=sub_graph.inf,
                            next_step=next_step,
                            loop=loop,
                        )

        return [x.output for x in sub_graphs]

    async def run_async(self, *args, inference_state: InferenceState, **kwargs):
        """
        Run through the operators using the provided router and scheduler.
        The input to a given operator is the output of the previous operator.

        :param inference_state: inference_state for the pipeline.
        :param pipeline_state: pipeline_state for the pipeline. The values in the state
            are created during pipeline creation and are read-only during inference.
        """
        loop = asyncio.get_running_loop()

        next_step = self.router.START_ROUTE
        operator_output = None

        while next_step != self.router.END_ROUTE:
            # Either a dictionary key or valid index

            if next_step == self.router.SPLIT_ROUTE:
                if operator_output is None:
                    raise ValueError(
                        f"{self.router.SPLIT_ROUTE} should appear after "
                        f"{self.ROUTER.START_ROUTE}"
                    )

                operator_output = await self._apply_split(
                    operator_output, inference_state, loop=loop
                )
                next_step = self.router.route[self.router.JOIN_ROUTE]
                if next_step == self.router.END_ROUTE:
                    return operator_output

            if next_step == self.router.START_ROUTE:
                outputs = run_func(
                    *args,
                    func=self._scheduler_group.submit,
                    operator=self.ops[next_step],
                    inference_state=inference_state,
                    pipeline_state=self.pipeline_state,
                    loop=loop,
                    **kwargs,
                )
                await outputs
                operator_output = outputs.result()

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
                state_update = operator_output[-1]
                operator_output = operator_output[0]

            next_step = self.router.next(next_step, self.ops, operator_output)
            if state_update:
                inference_state.update_state(state_update)
        return operator_output

    async def _apply_split(
        self,
        inp: Any,
        inference_state: InferenceState,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        batches, orig_batch_size = self.expand_inputs(inp, 1)

        # Create a list of SplitRoutes, per batch size 1
        # Each SplitRoute object holds information about the particular path it
        # follows. All start at the same step defined by SPLIT_ROUTE and start
        # with the same inference_state.
        split_graphs = [
            SubGraph(
                inf=copy.deepcopy(inference_state),
                step=self.router.route[self.router.SPLIT_ROUTE],
                end=[self.router.JOIN_ROUTE],
            )
            for i in range(len(batches))
        ]

        outputs = await self._run_sub_graphs(
            sub_graph_inputs=batches, sub_graphs=split_graphs, loop=loop
        )
        return self.condense_inputs(outputs)

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
            pipeline = Operator.create(task=task, **new_kwargs)
            if not isinstance(pipeline, cls):
                raise RuntimeError(
                    "Pipeline was not created for the given task. The "
                    "provided task should be registered using the OperatorRegistry"
                )
        except Exception:
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
        :param pipeline_state: pipeline_state for the pipeline. The values in the state
            are created during pipeline creation and are read-only during inference.
        """
        next_step = self.router.START_ROUTE
        operator_output = None
        while next_step != self.router.END_ROUTE:

            # Split Grap Execution (i.e multiple subgraphs)
            # NOTE: split_route should only appear after the start route node
            if next_step == self.router.SPLIT_ROUTE:
                if operator_output is None:
                    raise ValueError(
                        f"{self.router.SPLIT_ROUTE} should appear after "
                        f"{self.router.START_ROUTE}"
                    )

                operator_output = asyncio.run(
                    self._apply_split(operator_output, inference_state)
                )
                next_step = self.router.route[self.router.JOIN_ROUTE]
                if next_step == self.router.END_ROUTE:
                    return operator_output

            if next_step == self.router.START_ROUTE:
                operator_output = run_func(
                    *args,
                    func=self._scheduler_group.submit,
                    operator=self.ops[next_step],
                    inference_state=inference_state,
                    pipeline_state=self.pipeline_state,
                    **kwargs,
                ).result()

                if isinstance(operator_output, tuple):
                    operator_output, state_update = (
                        operator_output[0],
                        operator_output[-1],
                    )
                    inference_state.update_state(state_update)

                next_step = self.router.next(next_step, self.ops, operator_output)

            else:
                # Single graph execution
                graph = SubGraph(
                    inf=copy.deepcopy(inference_state),
                    step=next_step,
                    end=[self.router.SPLIT_ROUTE, self.router.END_ROUTE],
                )

                operator_output = asyncio.run(
                    self._run_sub_graphs(
                        sub_graph_inputs=[operator_output], sub_graphs=[graph]
                    )
                )[0]

                inference_state = graph.inf
                next_step = graph.step

        return operator_output

    def __call__(self, *args, **kwargs):
        """
        Consolidate any provided inference_state or pipeline_state objects and pass
        any other operator inputs to run().

        :return: output of the pipeline operators ran with the router for the given
            input
        """
        if kwargs.get("inference_state"):
            inference_state = kwargs.pop("inference_state")
        else:
            inference_state = InferenceState()
            inference_state.create_state({})

        kwargs["inference_state"] = inference_state

        return self.run(*args, **kwargs)

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


def text_generation_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    return Pipeline.create("text_generation", *args, **kwargs)


def code_generation_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    return Pipeline.create("code_generation", *args, **kwargs)


def chat_pipeline(*args, **kwargs) -> "Pipeline":
    """
    :return: text generation pipeline with the given args and
        kwargs passed to Pipeline.create
    """
    return Pipeline.create("chat", *args, **kwargs)


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
    """
    return Pipeline.create("information_retrieval_haystack", *args, **kwargs)


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
