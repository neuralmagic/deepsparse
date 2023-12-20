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
from typing import Dict, Optional

from deepsparse.pipeline import Pipeline
from deepsparse.routers import GraphRouter
from deepsparse.schedulers import OperatorScheduler
from deepsparse.transformers.helpers import setup_transformers_pipeline
from deepsparse.transformers.pipelines.text_generation import (
    CompileGenerations,
    GenerateNewTokenOperator,
    JoinOutput,
    NLEngineOperatorNoCache,
    ParseTextGenerationInputs,
    PrepareGeneration,
    ProcessInputsTextGeneration,
    ProcessOutputs,
    TokenGeneratorOperator,
)
from deepsparse.transformers.utils.helpers import process_generation_config
from deepsparse.utils import split_engine_inputs
from deepsparse.utils.onnx import default_cached_outputs


_LOGGER = logging.getLogger(__name__)


class TextGenerationPipelineNoCache(Pipeline):
    def __init__(
        self,
        model_path: str,
        sequence_length: int = 1024,
        onnx_model_name: Optional[str] = None,
        generation_config=None,
        engine_kwargs: Optional[Dict] = None,
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

        parse_inputs = ParseTextGenerationInputs()

        process_inputs = ProcessInputsTextGeneration(
            generation_config=process_generation_config(generation_config),
            sequence_length=sequence_length,
            tokenizer=self.tokenizer,
        )
        engine_operator = NLEngineOperatorNoCache(
            sequence_length=sequence_length,
            **engine_kwargs,
        )
        prepare_generation = PrepareGeneration(
            sequence_length=sequence_length,
            prompt_sequence_length=1,
            token_generator=token_generator,
        )
        generate_new_token = GenerateNewTokenOperator(
            tokenizer=self.tokenizer, force_max_tokens=True
        )
        compile_generations = CompileGenerations()
        join_output = JoinOutput(tokenizer=self.tokenizer)
        process_outputs = ProcessOutputs(tokenizer=self.tokenizer)

        ops = {
            "parse_inputs": parse_inputs,
            "process_input": process_inputs,
            "engine_operator": engine_operator,
            "prepare_generation": prepare_generation,
            "generate_new_token": generate_new_token,
            "compile_generations": compile_generations,
            "join_output": join_output,
            "process_outputs": process_outputs,
        }
        routes = {
            "parse_inputs": "process_input",
            "process_input": "SPLIT",
            "SPLIT": "engine_operator",
            "engine_operator": "prepare_generation",
            "prepare_generation": "generate_new_token",
            "generate_new_token": "compile_generations",
            "compile_generations": "JOIN",
            "JOIN": "join_output",
            "join_output": "process_outputs",
            "process_outputs": "STOP",
        }

        # TODO: Using the GraphRouter, but should use
        # LinearRouter with appropriate split/join support
        router = GraphRouter(
            end_route="STOP", start_route="process_input", route=routes
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops,
            router=router,
            schedulers=scheduler,
        )

    def run(self, *args, **kwargs):
        # we need to set the fixed_sequences_length flag to True
        # for the non-kv cache pipeline
        kwargs.update(dict(fixed_sequences_length=True, max_new_tokens=1))
        return super().run(*args, **kwargs)

    def condense_inputs(self, *args, **kwargs):
        return args[0], kwargs

    def expand_inputs(self, items, batch_size):
        items = [items.get(key) for key in items.keys()]
        out, orig_batch_size = split_engine_inputs(items, batch_size)
        combined_batches = [{"input_ids": b[0], "attention_mask": b[1]} for b in out]
        return combined_batches, orig_batch_size

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
