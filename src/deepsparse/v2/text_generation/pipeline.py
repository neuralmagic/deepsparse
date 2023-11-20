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

from typing import Dict, Optional

from deepsparse.transformers.helpers import setup_transformers_pipeline
from deepsparse.transformers.utils.helpers import process_generation_config
from deepsparse.utils import split_engine_inputs
from deepsparse.utils.onnx import default_cached_outputs
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import GraphRouter, LinearRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.text_generation import (
    AutoRegressiveOperatorPreprocess,
    CompileGeneratedTokens,
    CompileGenerations,
    CompilePromptLogits,
    GenerateNewTokenOperator,
    JoinOutput,
    KVCacheCreator,
    MultiEnginePrefill,
    NlEngineOperator,
    NlEngineOperatorNoCache,
    PrepareforPrefill,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    ProcessOutputs,
    TokenGeneratorOperator,
)
from deepsparse.v2.utils import PipelineState


class TextGenerationPipelineNoCache(Pipeline):
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 1024,
        engine_kwargs: Optional[Dict] = None,
        onnx_model_name: Optional[str] = None,
        generation_config=None,  # TODO: Typing here
        **kwargs,
    ):

        (
            self.model_path,
            self.config,
            self.tokenizer,
            engine_kwargs,
        ) = setup_transformers_pipeline(
            model_path,
            sequence_length,
            tokenizer_padding_side="right",
            onnx_model_name=onnx_model_name,
            engine_kwargs=engine_kwargs,
        )
        self.verify_no_kv_cache_present()

        token_generator = TokenGeneratorOperator()

        ops = [
            ProcessInputsTextGeneration(
                generation_config=process_generation_config(generation_config),
                sequence_length=sequence_length,
                tokenizer=self.tokenizer,
            ),
            NlEngineOperatorNoCache(**engine_kwargs),
            PrepareGeneration(
                sequence_length=sequence_length,
                prompt_sequence_length=1,
                token_generator=token_generator,
            ),
            GenerateNewTokenOperator(tokenizer=self.tokenizer, force_max_tokens=True),
            CompileGenerations(),
            JoinOutput(tokenizer=self.tokenizer),
            ProcessOutputs(tokenizer=self.tokenizer),
        ]
        router = LinearRouter(end_route=len(ops))
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops,
            router=router,
            schedulers=scheduler,
        )

    def run(self, *args, **kwargs):
        # we need to set the fixed_sequences_length flag to True
        # for the non-kv cache pipeline
        kwargs.update(dict(fixed_sequences_length=True))
        return super().run(*args, **kwargs)

    def verify_no_kv_cache_present(self) -> bool:
        """
        Verifies that the ONNX model does not have
        KV cache inputs/outputs present.
        :return: True if compatible, False otherwise
        """
        is_kv_cache_present = any(default_cached_outputs(self.model_path))
        if is_kv_cache_present:
            raise ValueError(
                f"The model: {self.model_path} has KV cache inputs/outputs present. "
                "Please use the TextGenerationPipeline instead."
            )
        return not is_kv_cache_present


class TextGenerationPipeline(Pipeline):
    def __init__(
        self,
        model_path: str,
        prompt_sequence_length: int = 16,
        sequence_length: int = 1024,
        internal_kv_cache: bool = True,
        force_max_tokens: bool = False,
        generation_config=None,
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

        single_engine_operator = NlEngineOperator(
            sequence_length=sequence_length,
            internal_kv_cache=internal_kv_cache,
            input_ids_length=1,
            **engine_kwargs,
        )

        multi_engine_operator = NlEngineOperator(
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
            ops=ops, router=router, schedulers=scheduler, pipeline_state=pipeline_state
        )

    def expand_inputs(self, items, batch_size):
        items = [items.get(key) for key in items.keys()]
        out, orig_batch_size = split_engine_inputs(items, batch_size)
        combined_batches = [{"input_ids": b[0], "attention_mask": b[1]} for b in out]
        return combined_batches, orig_batch_size

    def condense_inputs(self, *args, **kwargs):
        return args[0], kwargs
