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

from typing import Any, Dict, Optional

from deepsparse.transformers.utils.helpers import process_generation_config
from deepsparse.v2.routers import LinearRouter, TextGenerationRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.text_generation import (
    AutoRegressiveOperatorPreprocess,
    CompileGeneratedTokens,
    CompileGenerations,
    CompilePromptLogits,
    ComputeEngineInputs,
    GenerateNewTokenOperator,
    KVCacheCreator,
    NLEngineOperator,
    NLEngineOperatorNoCache,
    PrepareforPrefill,
    PrepareforSingleEngine,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    ProcessOutputs,
    TokenGeneratorOperator,
)
from deepsparse.v2.transformers.pipeline import TransformersPipeline
from deepsparse.v2.utils import PipelineState


__all__ = [
    "TextGenerationPipeline",
    "TextGenerationPipelineNoCache",
]


class TextGenerationPipelineNoCache(TransformersPipeline):
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 32,
        engine_kwargs: Optional[Dict] = None,
        onnx_model_name: Optional[str] = None,
    ):

        model_path = self.setup_onnx_file_path(
            model_path, sequence_length, onnx_model_name=onnx_model_name
        )
        self.check_if_compatible(model_path)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if engine_kwargs is None:
            engine_kwargs = {}
        engine_kwargs["model_path"] = model_path

        token_generator = TokenGeneratorOperator()

        # Setup the operators of this pipeline
        ops = [
            ProcessInputsTextGeneration(
                sequence_length=sequence_length, tokenizer=self.tokenizer
            ),
            NLEngineOperatorNoCache(sequence_length=sequence_length, **engine_kwargs),
            ProcessOutputs(tokenizer=self.tokenizer),
        ]

        router = LinearRouter(route=ops)
        scheduler = [OperatorScheduler()]
        super().__init__(ops=ops, router=router, schedulers=scheduler)

    def run(self, inp: Any, **kwargs):
        inp.update(dict(fixed_sequences_length=True))
        out, _ = super().run(inp, **kwargs)
        return out, {}

    @staticmethod
    def check_if_compatible(model_path: str) -> bool:
        """
        Check if the model is compatible with the pipeline

        :param model_path: path to the model
        :return: True if compatible, False otherwise
        """
        pass


class TextGenerationPipeline(TransformersPipeline):
    def __init__(
        self,
        model_path: str,
        prompt_sequence_length: int = 16,
        sequence_length: int = 1024,
        internal_kv_cache: bool = True,
        force_max_tokens: bool = False,
        generation_config=None,
        engine_kwargs: Dict = None,
    ):

        pipeline_state = PipelineState()
        pipeline_state_vals = {}

        # TODO: The code below will be replaced with a transformers set-up Operator.
        model_path = self.setup_onnx_file_path(model_path, sequence_length)
        self.tokenizer.padding_side = "left"
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not engine_kwargs:
            engine_kwargs = {}
        engine_kwargs["model_path"] = model_path

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
        ] = multi_engine_operator.onnx_input_names_no_cache
        pipeline_state_vals["cache_shape"] = multi_engine_operator.cache_shape
        pipeline_state_vals["output_names"] = multi_engine_operator.output_names
        pipeline_state_vals[
            "kv_cache_data_type"
        ] = multi_engine_operator.kv_cache_data_type
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
        prep_for_single_engine = PrepareforSingleEngine(
            prompt_sequence_length=prompt_sequence_length,
            sequence_length=sequence_length,
        )
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

        ops = {
            "process_input": process_inputs,
            "single_engine": single_engine_operator,
            "multi_engine": multi_engine_operator,
            "kv_cache_creator": kv_cache_creator,
            "prepare_prefill": engine_inputs_for_prefill,
            "multi_engine_prefill": multi_engine_prefill,
            "compile_logits": compile_prompt_logits,
            "prepare_single_engine": prep_for_single_engine,
            "autoregressive_preprocess": autoregressive_preprocess,
            "prep_for_generation": prep_for_generation,
            "generate_new_token": generate_new_token,
            "process_outputs": process_output,
            "compile_generations": compile_generations,
            "compile_generated_tokens": compile_generated_tokens,
        }

        routes = {
            "process_input": "prepare_prefill",
            "prepare_prefill": ["multi_engine_prefill", "prepare_single_engine"],
            "multi_engine_prefill": "multi_engine",
            "multi_engine": "compile_logits",
            "compile_logits": [
                "multi_engine_prefill",
                "prepare_single_engine",
                "autoregressive_preprocess",
                "prep_for_generation",
            ],
            "prepare_single_engine": [
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
            "compile_generations": "process_outputs",
            "process_outputs": "STOP",
        }

        router = TextGenerationRouter(
            end_route="STOP", start_route="process_input", route=routes
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops, router=router, schedulers=scheduler, pipeline_state=pipeline_state
        )
