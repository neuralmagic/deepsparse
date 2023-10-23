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

from typing import Dict

from deepsparse.transformers.utils.helpers import process_generation_config
from deepsparse.v2.operators import Operator
from deepsparse.v2.pipeline import Pipeline
from deepsparse.v2.routers import TextGenerationRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.text_generation import (
    CompilePromptLogits,
    KVCacheCreator,
    MultiEnginePrefill,
    NLEngineOperator,
    PrepareforMultiEngine,
    PrepareforPrefill,
    ProcessInputsTextGeneration,
    TokensToEngineInputs,
)
from deepsparse.v2.utils import PipelineState


class DoNothing(Operator):
    def run():
        return


class TextGenerationPipeline(Pipeline):
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

        # transformers_preprocess = TransformersPreprocess() ## set-up config/tokenizer ## should give us a tokenizer, onnxfilepath
        # temporarily copy/pasta transformers code until this operator is set-up
        self.tokenizer = None
        model_path = self.setup_onnx_file_path(model_path, sequence_length)

        if not engine_kwargs:
            engine_kwargs = {}
        engine_kwargs["model_path"] = model_path

        if internal_kv_cache and engine_kwargs.get("engine_type") == "onnxruntime":
            internal_kv_cache = False

        single_engine_operator = NLEngineOperator(
            sequence_length=sequence_length,
            internal_kv_cache=internal_kv_cache,
            input_ids_length=prompt_sequence_length,
            **engine_kwargs,
        )

        multi_engine_operator = NLEngineOperator(
            sequence_length=sequence_length,
            internal_kv_cache=internal_kv_cache,
            input_ids_length=1,
            **engine_kwargs,
        )

        pipeline_state_vals[
            "onnx_input_names_no_cache"
        ] = single_engine_operator.onnx_input_names_no_cache
        pipeline_state_vals["cache_shape"] = single_engine_operator.cache_shape
        pipeline_state_vals["output_names"] = single_engine_operator.output_names
        pipeline_state_vals[
            "kv_cache_data_type"
        ] = single_engine_operator.kv_cache_data_type
        pipeline_state.create_state(pipeline_state_vals)
        print(pipeline_state_vals)

        # Can have the transformers call the operator inside this operator --> need tokenzier and model_path, fed into engine
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

        # Operators has a dependency on the engine_operator as depends on the engine
        # We can either initialize or we store everything in a pipeline state (this couples the operators together)
        tokens_to_engine_input = TokensToEngineInputs()
        engine_inputs_for_prefill = PrepareforPrefill(kv_cache_creator=kv_cache_creator)
        prepare_for_multi_engine = PrepareforMultiEngine(
            prompt_sequence_length=prompt_sequence_length
        )
        multi_engine_prefill = MultiEnginePrefill(
            prompt_sequence_length=prompt_sequence_length,
            sequence_length=sequence_length,
        )
        compile_prompt_logits = CompilePromptLogits()
        do_nothing = DoNothing()

        ops = {
            "process_input": process_inputs,
            "single_engine": single_engine_operator,
            "multi_engine": multi_engine_operator,
            "kv_cache_creator": kv_cache_creator,
            "tokens_to_engine": tokens_to_engine_input,
            "prepare_prefill": engine_inputs_for_prefill,
            "prepare_multiengine": prepare_for_multi_engine,
            "multi_engine_prefill": multi_engine_prefill,
            "do_nothing": do_nothing,
            "compile_prompt_logits": compile_prompt_logits,
        }

        routes = {
            "process_input": "tokens_to_engine",
            "tokens_to_engine": "prepare_prefill",
            "prepare_prefill": ["prepare_multiengine", "do_nothing"],
            "prepare_multiengine": "multi_engine_prefill",
            "multi_engine_prefill": "multi_engine",
            "multi_engine": "compile_logits",
            "compile_logits": ["prepare_multiengine", "do_nothing"],
            "do_nothing": "STOP",
        }

        router = TextGenerationRouter(
            end_route="STOP", start_route="process_input", route=routes
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops, router=router, schedulers=scheduler, pipeline_state=pipeline_state
        )

    # stealing this for now
    def setup_onnx_file_path(self, model_path, sequence_length) -> str:
        import logging

        import transformers
        from transformers import AutoTokenizer

        from deepsparse.transformers.helpers import get_deployment_path

        """
        Parses ONNX model from the `model_path` provided. It additionally
        creates config and tokenizer objects from the `deployment path`,
        derived from the `model_path` provided.

        :return: file path to the processed ONNX file for the engine to compile
        """
        deployment_path, onnx_path = get_deployment_path(model_path)

        # temporarily set transformers logger to ERROR to avoid
        # printing misleading warnings
        hf_logger = logging.getLogger("transformers")
        hf_logger_level = hf_logger.level
        hf_logger.setLevel(logging.ERROR)
        self.config = transformers.PretrainedConfig.from_pretrained(
            deployment_path,
            finetuning_task=self.task if hasattr(self, "task") else None,
        )
        hf_logger.setLevel(hf_logger_level)

        self._trust_remote_code = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            deployment_path,
            trust_remote_code=self._trust_remote_code,
            model_max_length=sequence_length,
        )

        if not self.config or not self.tokenizer:
            raise RuntimeError(
                "Invalid config or tokenizer provided. Please provide "
                "paths to the files or ensure they exist in the `model_path` provided. "
                "See `tokenizer` and `config` arguments for details."
            )
        return onnx_path
