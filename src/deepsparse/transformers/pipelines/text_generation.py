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

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse import Context, MultiModelEngine
from deepsparse.pipeline import (
    DEEPSPARSE_ENGINE,
    ORT_ENGINE,
    SUPPORTED_PIPELINE_ENGINES,
    Engine,
    ORTEngine,
)
from deepsparse.transformers.helpers import overwrite_transformer_onnx_model_inputs
from deepsparse.transformers.pipelines import TransformersPipeline


_MODEL_DIR_ONNX_DECODER_NAME = "decoder_model.onnx"
from transformers import BatchEncoding

from deepsparse import Pipeline


__all__ = ["TextGenerationPipeline"]


class InputSchema(BaseModel):
    sequences: Union[str, List[str]]


class OutputSchema(BaseModel):
    sequences: Union[str, List[str]]


@Pipeline.register(
    task="text_generation",
    task_aliases=["codegen"],
)
class TextGenerationPipeline(TransformersPipeline):
    @staticmethod
    def route_input_to_bucket(
        *args, input_schema: BaseModel, pipelines: List[Pipeline], **kwargs
    ) -> Pipeline:
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.onnx_decoder_path = self.setup_decoder_onnx_file_path()
        self.decoder_engine = self._initialize_decoder_engine()

    @property
    def input_schema(self) -> Type[BaseModel]:
        return InputSchema

    @property
    def output_schema(self) -> Type[BaseModel]:
        return OutputSchema

    def process_engine_outputs(
        self, engine_outputs: List[numpy.ndarray], **kwargs
    ) -> BaseModel:
        return None

    def process_inputs(self, inputs: BaseModel):

        sequences = inputs.sequences
        if isinstance(sequences, List) and all(
            isinstance(sequence, List) and len(sequence) == 1 for sequence in sequences
        ):
            # if batch items contain only one sequence but are wrapped in lists, unwrap
            # for use as tokenizer input
            sequences = [sequence[0] for sequence in sequences]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        input_tokens = self.tokenizer(
            sequences,
            add_special_tokens=True,
            return_tensors="np",
            max_length=self.sequence_length,
            padding="max_length",
        )

        onnx_input_names = [
            input_name
            for input_name in self.onnx_input_names
            if not input_name.startswith("past_key_values")
        ]
        engine_input = self.tokens_to_engine_input(
            input_tokens, onnx_input_names=onnx_input_names
        )

        # a boolean mask that indicates which tokens are valid (are non-padding tokens)
        valid_tokens_mask = numpy.where(
            engine_input[0] == self.tokenizer.pad_token_id, 1, 0
        )

        preprocessing_kwargs = dict(
            input_sequence=engine_input[0], valid_tokens_mask=valid_tokens_mask
        )

        return engine_input, preprocessing_kwargs

    def engine_forward(
        self, engine_inputs: List[numpy.ndarray], **kwargs
    ) -> List[numpy.ndarray]:
        """
        :param engine_inputs: list of numpy inputs to Pipeline engine forward
            pass
        :return: result of forward pass to Pipeline engine
        """
        assert self._batch_size == 1

        eos_token_found = False
        generated_tokens = []
        valid_tokens = [
            t for t in engine_inputs[0][0] if t != self.tokenizer.pad_token_id
        ]

        past_logits, *new_kvs = self.decoder_engine(engine_inputs)

        new_token = numpy.argmax(past_logits[0, -1, :])
        generated_tokens.append(new_token)

        kv_output_names = [
            name
            for name in self.decoder_engine._output_names
            if name.startswith("present")
        ]
        kv_cache = dict(zip(kv_output_names, new_kvs))
        kv_cache = {
            k.replace("present", "past_key_values"): v[:, :, :-1]
            for k, v in kv_cache.items()
        }
        for k, v in kv_cache.items():
            v[:, :, len(valid_tokens) :] = 0.0

        for iter in range(self.sequence_length - len(valid_tokens)):
            if eos_token_found:
                return valid_tokens

            attention_mask = numpy.zeros((1, self.sequence_length), dtype=numpy.int64)
            attention_mask[:, : len(valid_tokens)] = 1
            attention_mask[:, -1] = 1
            assert attention_mask.sum() == len(valid_tokens) + 1

            engine_inputs_dict = {
                "input_ids": numpy.array([[new_token]]),
                "attention_mask": attention_mask,
            }
            engine_inputs_dict.update(kv_cache)
            engine_inputs = [
                numpy.ascontiguousarray(engine_inputs_dict[name])
                for name in self.onnx_input_names
            ]

            new_logits, *new_kvs = self.engine(engine_inputs)

        return engine_inputs[0]

    def setup_decoder_onnx_file_path(self):
        decoder_onnx_path = os.path.join(self.model_path, _MODEL_DIR_ONNX_DECODER_NAME)
        (
            decoder_onnx_path,
            self.decoder_onnx_input_names,
            self._temp_model_directory,
        ) = overwrite_transformer_onnx_model_inputs(
            decoder_onnx_path, max_length=self.sequence_length
        )

        return decoder_onnx_path

    def _initialize_decoder_engine(self) -> Union[Engine, ORTEngine]:
        engine_type = self.engine_type.lower()

        if engine_type == DEEPSPARSE_ENGINE:
            if self.context is not None and isinstance(self.context, Context):
                self._engine_args.pop("num_cores", None)
                self._engine_args.pop("scheduler", None)
                self._engine_args["context"] = self.context
                return MultiModelEngine(
                    model=self.onnx_decoder_path,
                    **self._engine_args,
                )
            return Engine(self.onnx_decoder_path, **self._engine_args)
        elif engine_type == ORT_ENGINE:
            return ORTEngine(self.onnx_decoder_path, **self._engine_args)
        else:
            raise ValueError(
                f"Unknown engine_type {self.engine_type}. Supported values include: "
                f"{SUPPORTED_PIPELINE_ENGINES}"
            )
