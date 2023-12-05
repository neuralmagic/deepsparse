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
from typing import Dict, List, Optional

from deepsparse.transformers.helpers import setup_transformers_pipeline
from deepsparse.transformers.utils.helpers import process_generation_config
from deepsparse.utils import split_engine_inputs
from deepsparse.v2.operators import EngineOperator
from deepsparse.v2.operators.registry import OperatorRegistry
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import GraphRouter
from deepsparse.v2.schedulers import ContinuousBatchingScheduler, OperatorScheduler
from deepsparse.v2.text_generation import (
    AutoRegressiveOperatorPreprocess,
    CompileGeneratedTokens,
    CompileGenerations,
    CompilePromptLogits,
    GenerateNewTokenOperator,
    JoinOutput,
    KVCacheCreator,
    MultiEnginePrefill,
    NLEngineOperator,
    PrepareforPrefill,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    ProcessOutputs,
    TokenGeneratorOperator,
)
from deepsparse.v2.utils import PipelineState


_LOGGER = logging.getLogger(__name__)


@OperatorRegistry.register(name="text_generation")
class TextGenerationPipeline(Pipeline):
    def __init__(
        self,
        model_path: str,
        prompt_sequence_length: int = 16,
        sequence_length: int = 1024,
        internal_kv_cache: bool = True,
        force_max_tokens: bool = False,
        generation_config=None,
        continuous_batch_sizes: Optional[List[int]] = None,
        engine_kwargs: Optional[Dict] = None,
    ):
        (
            self.model_path,
            self.config,
            self.tokenizer,
            engine_kwargs,
        ) = setup_transformers_pipeline(
            model_path, sequence_length, engine_kwargs=engine_kwargs
        )

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
        prep_for_generation = PrepareGeneration(
            sequence_length=sequence_length,
            prompt_sequence_length=prompt_sequence_length,
            token_generator=token_generator,
        )
        generate_new_token = GenerateNewTokenOperator(
            tokenizer=self.tokenizer, force_max_tokens=force_max_tokens
        )
        process_output = ProcessOutputs(tokenizer=self.tokenizer)
        compile_generations = CompileGenerations()
        compile_generated_tokens = CompileGeneratedTokens()
        join_output = JoinOutput(tokenizer=self.tokenizer)

        # TODO: do we want to support lists for different engines?
        continuous_batching_scheduler = None
        if continuous_batch_sizes:
            if internal_kv_cache:
                _LOGGER.warn(
                    "internal kv_cache is not supported with continuous_batching "
                )
            else:
                continuous_batching_scheduler = self._get_continuous_batching_scheduler(
                    batch_sizes=continuous_batch_sizes,
                    engines=[single_engine_operator, multi_engine_operator],
                )

        ops = {
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
        }

        routes = {
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
            "prep_for_generation": "autoregressive_preprocess",
            "generate_new_token": "compile_generated_tokens",
            "compile_generated_tokens": [
                "autoregressive_preprocess",
                "compile_generations",
            ],
            "compile_generations": "JOIN",
            "JOIN": "join_output",
            "join_output": "process_outputs",
            "process_outputs": "STOP",
        }

        router = GraphRouter(
            end_route="STOP", start_route="process_input", route=routes
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops,
            router=router,
            schedulers=scheduler,
            pipeline_state=pipeline_state,
            continuous_batching_scheduler=continuous_batching_scheduler,
        )

    def expand_inputs(self, items, batch_size):
        items = [items.get(key) for key in items.keys()]
        out, orig_batch_size = split_engine_inputs(items, batch_size)
        combined_batches = [{"input_ids": b[0], "attention_mask": b[1]} for b in out]
        return combined_batches, orig_batch_size

    def condense_inputs(self, *args, **kwargs):
        return args[0], kwargs

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
