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
from deepsparse.v2.routers import GraphRouter
from deepsparse.v2.schedulers import OperatorScheduler
from deepsparse.v2.text_generation import (
    AutoRegressiveOperatorPreprocess,
    CompilePromptLogits,
    KVCacheCreator,
    MultiEnginePrefill,
    NLEngineOperator,
    PrepareforPrefill,
    PrepareforSingleEngine,
    ProcessInputsTextGeneration,
)
from deepsparse.v2.utils import PipelineState


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

        # TODO: The code below will be replaced with a transformers set-up Operator.
        self.tokenizer = None
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
        # attributes to the specific Operator that neeed them, as class attributes.
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
        """
        prep_for_single_engine = PrepareforSingleEngine(
            prompt_sequence_length=prompt_sequence_length,
            sequence_length=sequence_length,
        )
        """
        autoregressive_preprocess = AutoRegressiveOperatorPreprocess(
            sequence_length=sequence_length,
            prompt_sequence_length=prompt_sequence_length,
        )
        final_step = FinalStep()

        ops = {
            "process_input": process_inputs,
            "single_engine": single_engine_operator,
            "multi_engine": multi_engine_operator,
            "kv_cache_creator": kv_cache_creator,
            "prepare_prefill": engine_inputs_for_prefill,
            "multi_engine_prefill": multi_engine_prefill,
            "compile_logits": compile_prompt_logits,
            "autoregressive_preprocess": autoregressive_preprocess,
            "final_step": final_step,
        }

        routes = {
            "process_input": "prepare_prefill",
            "prepare_prefill": ["multi_engine_prefill", "autoregressive_preprocess"],
            "multi_engine_prefill": "multi_engine",
            "multi_engine": "compile_logits",
            "compile_logits": [
                "multi_engine_prefill",
                "autoregressive_preprocess",
                "final_step",
            ],
            "autoregressive_preprocess": "single_engine",
            "single_engine": "compile_logits",
            "final_step": "STOP",
        }

        router = GraphRouter(
            end_route="STOP", start_route="process_input", route=routes
        )
        scheduler = [OperatorScheduler()]
        super().__init__(
            ops=ops, router=router, schedulers=scheduler, pipeline_state=pipeline_state
        )

    # TODO: Move to be part of a generic transformers set-up Operator.
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


# NOTE: This is a dummy last step which will be removed. Used as a final step
# for the current routes.
class FinalStep(Operator):
    def can_operate(self, *args, **kwargs):
        return True

    def run(self, *args, **kwargs):
        import numpy

        inference_state = kwargs.get("inference_state")
        prompt_logits = inference_state.current_state.get("prompt_logits")
        return numpy.concatenate(prompt_logits, axis=1)
