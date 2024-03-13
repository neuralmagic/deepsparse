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

import logging
from typing import List, Optional

from deepsparse.loggers.logger_manager import LoggerManager
from deepsparse.middlewares import MiddlewareManager
from deepsparse.operators import EngineOperator
from deepsparse.operators.registry import OperatorRegistry
from deepsparse.pipeline import Pipeline
from deepsparse.routers import GraphRouter
from deepsparse.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from deepsparse.transformers.helpers import setup_transformers_pipeline
from deepsparse.transformers.pipelines.text_generation import (
    AutoRegressiveOperatorPreprocess,
    CompileGeneratedTokens,
    CompileGenerations,
    CompilePromptLogits,
    GenerateNewTokenOperator,
    JoinOutput,
    KVCacheCreator,
    MultiEnginePrefill,
    NLEngineOperator,
    ParseTextGenerationInputs,
    PrepareforPrefill,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    ProcessOutputs,
    ProcessStreamingOperator,
    TokenGeneratorOperator,
)
from deepsparse.transformers.pipelines.text_generation.pipeline_no_kv_cache import (
    TextGenerationPipelineNoCache,
)
from deepsparse.transformers.utils.helpers import (
    causal_mask_input_present,
    process_generation_config,
    verify_kv_cache_present,
)
from deepsparse.utils import PipelineState, split_engine_inputs


_LOGGER = logging.getLogger(__name__)


@OperatorRegistry.register(name="text_generation", alias=["opt", "mpt", "llama"])
class TextGenerationPipeline(Pipeline):
    DEFAULT_SEQUENCE_LENGTH = 1024

    def __new__(
        cls,
        model_path: str,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        onnx_model_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Dynamically decide which pipeline (with KV Cache support or without)
        to initialize based on the model_path
        (check whether it has a KV Cache inputs/outputs or not)

        :param model_path: path to the model to use for text generation.
        :param sequence_length: sequence length to compile model and tokenizer for.
            This controls the maximum context length of the pipeline. Default is 1024
        :param onnx_model_name: name of the onnx model to use for text generation.
            This is only used if the model_path is a directory. If None, defaults to
            model.onnx
        """
        model_path, config, tokenizer = setup_transformers_pipeline(
            model_path=model_path,
            sequence_length=sequence_length,
            onnx_model_name=onnx_model_name,
        )

        if not verify_kv_cache_present(model_path):
            _LOGGER.info(
                "Initializing TextGenerationPipeline without KV Cache support. "
                "Some of the input parameters will be ignored."
            )
            tokenizer.padding_side = "right"
            return TextGenerationPipelineNoCache(
                model_path=model_path,
                config=config,
                tokenizer=tokenizer,
                sequence_length=sequence_length,
                **kwargs,
            )

        # because we route variables to the object that implements
        # __new__, instead of passing newly created variables through
        # __init__, we need to store the variables in the class
        cls.model_path = model_path
        cls.config = config
        cls.tokenizer = tokenizer

        return super().__new__(cls)

    def __init__(
        self,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        prompt_sequence_length: int = 16,
        internal_kv_cache: bool = True,
        force_max_tokens: bool = False,
        continuous_batch_sizes: Optional[List[int]] = None,
        generation_config=None,
        benchmark: bool = False,
        middleware_manager: Optional[MiddlewareManager] = None,
        logger_manager: Optional[LoggerManager] = None,
        **engine_kwargs,
    ):
        """
        Pipeline for text generation tasks.

        :param sequence_length: sequence length to compile model and tokenizer for.
            This controls the maximum context length of the pipeline. Default is 1024
        :param prompt_sequence_length: For large prompts, the prompt is
            processed in chunks of this length. This is to maximize the inference
            speed. By default, this is set to 16. The length should also be 1 or a
            multiple of 4.
        :param internal_kv_cache: if True, the pipeline will use the deepsparse kv cache
            for caching the model outputs.
        :param force_max_tokens: if True, the pipeline will generate the maximum number
            of tokens supplied even if the stop token is reached.
        :param continuous_batch_sizes: Batch sizes to use for the continuous batching
            scheduler. If this scheduler is desired, a list of batch sizes can be
            provided. Each batch size must be a power of 2.
        :param generation_config: config file consisting of parameters used to control
            sequences generated for each prompt. The current supported parameters are:
            max_length, max_new_tokens, num_return_sequences, output_scores, top_p,
            top_k, repetition_penalty, do_sample, temperature. If None is provided,
            deepsparse defaults will be used. For all other input types, HuggingFace
            defaults for GenerationConfig will be used.
        :param benchmark: if True, the pipeline will print out the time taken for each
            operator.
        :param middleware_manager: MiddlewareManager object to use for the pipeline.
        :param engine_kwargs: kwargs for the engine. These will be used to initialize
            the EngineOperator.
        """
        # potentially pop out the onnx_model_name from engine_kwargs if it exists
        # (necessary for due to the existence of __new__)
        engine_kwargs.pop("onnx_model_name", None)

        if (prompt_sequence_length % 4 != 0) and (prompt_sequence_length != 1):
            raise ValueError(
                f"prompt_sequence_length must be 1 or multiple of 4. "
                f"prompt_sequence_length is {prompt_sequence_length}"
            )

        causal_mask_present = causal_mask_input_present(self.model_path)
        if not causal_mask_present:
            _LOGGER.warning(
                "This ONNX graph does not support processing the prompt"
                "with processing length > 1. Setting prompt_sequence_length to 1."
            )
            prompt_sequence_length = 1

        pipeline_state = PipelineState()
        pipeline_state_vals = {}

        if internal_kv_cache and engine_kwargs.get("engine_type") == "onnxruntime":
            internal_kv_cache = False

        single_engine_operator = NLEngineOperator(
            sequence_length=sequence_length,
            internal_kv_cache=internal_kv_cache,
            input_ids_length=1,
            **engine_kwargs,
        )

        multi_engine_operator = NLEngineOperator(
            sequence_length=sequence_length,
            internal_kv_cache=internal_kv_cache,
            input_ids_length=prompt_sequence_length,
            **engine_kwargs,
        )

        # NOTE: Currently using pipeline state. Can swap to simply pass in the
        # attributes to the specific Operator that need them, as class attributes.
        pipeline_state_vals[
            "onnx_input_names_no_cache"
        ] = single_engine_operator.onnx_input_names_no_cache
        pipeline_state_vals["cache_shape"] = single_engine_operator.cache_shape
        pipeline_state_vals["output_names"] = single_engine_operator.output_names
        pipeline_state_vals[
            "kv_cache_data_type"
        ] = single_engine_operator.kv_cache_data_type
        pipeline_state.create_state(pipeline_state_vals)

        parse_inputs = ParseTextGenerationInputs()
        process_inputs = ProcessInputsTextGeneration(
            generation_config=process_generation_config(generation_config),
            sequence_length=sequence_length,
            tokenizer=self.tokenizer,
        )

        kv_cache_creator = KVCacheCreator(
            sequence_length=sequence_length,
            tokenizer=self.tokenizer,
            prompt_sequence_length=prompt_sequence_length,
            internal_kv_cache=internal_kv_cache,
        )

        # NOTE: Can also have the KVCacheCreator be initialized inside this Operator.
        # Relies on pipeline state variables set-up above (can be swapped to be class
        # attributes instead of using the state.
        engine_inputs_for_prefill = PrepareforPrefill(kv_cache_creator=kv_cache_creator)

        multi_engine_prefill = MultiEnginePrefill(
            prompt_sequence_length=prompt_sequence_length,
            sequence_length=sequence_length,
        )
        compile_prompt_logits = CompilePromptLogits()

        autoregressive_preprocess = AutoRegressiveOperatorPreprocess(
            sequence_length=sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )
        token_generator = TokenGeneratorOperator()

        generate_new_token = GenerateNewTokenOperator(
            tokenizer=self.tokenizer, force_max_tokens=force_max_tokens
        )
        process_output = ProcessOutputs(tokenizer=self.tokenizer)
        compile_generations = CompileGenerations()
        compile_generated_tokens = CompileGeneratedTokens()
        join_output = JoinOutput(tokenizer=self.tokenizer)
        process_streaming_output = ProcessStreamingOperator(tokenizer=self.tokenizer)
        prep_for_generation = PrepareGeneration(
            sequence_length=sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            token_generator=token_generator,
        )

        # TODO: do we want to support lists for different engines?
        continuous_batching_scheduler = None
        if continuous_batch_sizes:
            continuous_batching_scheduler = self._get_continuous_batching_scheduler(
                batch_sizes=continuous_batch_sizes,
                engines=[single_engine_operator, multi_engine_operator],
            )

        ops = {
            "parse_inputs": parse_inputs,
            "process_input": process_inputs,
            "single_engine": single_engine_operator,
            "multi_engine": multi_engine_operator,
            "kv_cache_creator": kv_cache_creator,
            "prepare_prefill": engine_inputs_for_prefill,
            "multi_engine_prefill": multi_engine_prefill,
            "compile_logits": compile_prompt_logits,
            "autoregressive_preprocess": autoregressive_preprocess,
            "prep_for_generation": prep_for_generation,
            "generate_new_token": generate_new_token,
            "process_outputs": process_output,
            "compile_generations": compile_generations,
            "compile_generated_tokens": compile_generated_tokens,
            "join_output": join_output,
            "streaming_outputs": process_streaming_output,
        }

        base_routes = {
            "parse_inputs": "process_input",
            "process_input": "SPLIT",
            "SPLIT": "prepare_prefill",
            "prepare_prefill": ["multi_engine_prefill", "autoregressive_preprocess"],
            "multi_engine_prefill": "multi_engine",
            "multi_engine": "compile_logits",
            "compile_logits": [
                "multi_engine_prefill",
                "prep_for_generation",
                "autoregressive_preprocess",
            ],
            "autoregressive_preprocess": "single_engine",
            "single_engine": [
                "compile_logits",
                "generate_new_token",
            ],
            "prep_for_generation": "generate_new_token",
            "generate_new_token": "compile_generated_tokens",
        }

        routes = {
            "compile_generated_tokens": [
                "autoregressive_preprocess",
                "compile_generations",
            ],
            "compile_generations": "JOIN",
            "JOIN": "join_output",
            "join_output": "process_outputs",
            "process_outputs": "STOP",
        }

        streaming_route = {
            "streaming_outputs": ["autoregressive_preprocess", "JOIN"],
            "compile_generated_tokens": "streaming_outputs",
            "JOIN": "STOP",
        }

        routes.update(base_routes)
        streaming_route.update(base_routes)

        router = GraphRouter(
            end_route="STOP",
            start_route="parse_inputs",
            route=routes,
        )

        generator_router = GraphRouter(
            end_route="STOP", start_route="parse_inputs", route=streaming_route
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops,
            router=router,
            generator_router=generator_router,
            schedulers=scheduler,
            pipeline_state=pipeline_state,
            continuous_batching_scheduler=continuous_batching_scheduler,
            middleware_manager=middleware_manager,
            logger_manager=logger_manager,
            benchmark=benchmark,
        )

    @property
    def input_schema(self):
        return self.ops["process_input"].input_schema

    @property
    def output_schema(self):
        return self.ops["process_outputs"].output_schema

    def expand_inputs(self, items, batch_size):
        items = [items.get(key) for key in items.keys()]
        out, orig_batch_size = split_engine_inputs(items, batch_size)
        combined_batches = [{"input_ids": b[0], "attention_mask": b[1]} for b in out]
        return combined_batches, orig_batch_size

    def condense_inputs(self, *args, **kwargs):
        return args[0], kwargs

    @property
    def sequence_length(self) -> int:
        """
        Property to return the sequence length for the pipeline.
        (relies on the single engine operator)

        :return: the sequence length for the pipeline
        """
        return self.ops["single_engine"].sequence_length

    @property
    def batch_size(self) -> int:
        return self.ops["single_engine"].batch_size

    @property
    def engine_type(self) -> str:
        return self.ops["single_engine"]._engine_type

    def _get_continuous_batching_scheduler(
        self, batch_sizes: List[int], engines: List[EngineOperator]
    ) -> ContinuousBatchingScheduler:
        """
        Fetch the continuous batching scheduler. Requires adding the EngineOperator
        that will run through the scheduler.

        :param batch_sizes: List of batch sizes to be used by the models
        :param engine: List of EngineOperators which should be scheduled using the
            continuous batching scheduler

        :returns: ContinuousBatchingScheduler
        """
        continuous_batching_scheduler = ContinuousBatchingScheduler.get_instance()
        for op in engines:
            continuous_batching_scheduler.add_engine_operator(op, batch_sizes)
        return continuous_batching_scheduler
