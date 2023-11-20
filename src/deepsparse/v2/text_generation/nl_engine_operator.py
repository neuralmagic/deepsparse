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

import copy
import os
from typing import Any, List, Tuple

import numpy
from pydantic import BaseModel, Field

from deepsparse.utils.onnx import (
    CACHE_INPUT_PREFIX,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)
from deepsparse.v2.operators.engine_operator import (
    DEEPSPARSE_ENGINE,
    EngineOperator,
    EngineOperatorInputs,
)


__all__ = [
    "NlEngineOperator",
    "NlEngineOperatorNoCache",
    "NlEngineInputNoCache",
    "NlEngineInput",
]


class NlEngineInput(BaseModel):
    engine_inputs: List = Field(description="engine inputs")
    kv_cache: Any = Field(description="kv_cache object")
    tokens: List = Field(description="tokens")
    in_generation: bool = Field(description="in_generation", default=None)


class NlEngineInputNoCache(BaseModel):
    input_ids: Any
    attention_mask: Any


class NlEngineOperator(EngineOperator):

    """
    Operator for the NL Decoder Engine. This Operator inherits from the EngineOperator.
    Specific updates to engine attributes are made through this operator, as well
    as updating the kv_cache. This Operator is used for both the single-token and
    multi-token case.
    """

    input_schema = NlEngineInput
    output_schema = None

    def __init__(
        self,
        sequence_length: int,
        input_ids_length: int,
        internal_kv_cache: bool = False,
        **kwargs,
    ):

        self.kv_cache_data_type = None
        (
            onnx_file_path,
            output_indices_to_be_cached,
            kv_cache_data_type,
        ) = overwrite_onnx_model_inputs_for_kv_cache_models(
            onnx_file_path=kwargs.get("model_path"),
            batch_size=kwargs.get("batch_size", 1),
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )

        engine_kwargs = kwargs.get("engine_kwargs", {})
        if kwargs.get("engine_type", DEEPSPARSE_ENGINE) == DEEPSPARSE_ENGINE:
            if "WAND_OPT_FLAGS" not in os.environ:
                os.environ["WAND_OPT_FLAGS"] = "default,~pyramids"

        if any(output_indices_to_be_cached):
            self.kv_cache_data_type = kv_cache_data_type
            if (
                internal_kv_cache
                and kwargs.get("engine_type", DEEPSPARSE_ENGINE) == DEEPSPARSE_ENGINE
            ):
                engine_kwargs["cached_outputs"] = output_indices_to_be_cached

        kwargs["engine_kwargs"] = engine_kwargs
        kwargs["model_path"] = onnx_file_path
        super().__init__(**kwargs)

        self.input_ids_length = input_ids_length

    def run(self, inp: NlEngineInput, **kwargs) -> Any:
        engine_input = inp.engine_inputs
        kv_cache = inp.kv_cache

        inputs = self._add_kv_cache_to_input(engine_input, kv_cache)
        if bool(kv_cache.engine_internal_cache):
            # conventionally, before dispatching
            # inputs to the engine, we validate them
            # if val_inp=True. However, in this case
            # we want to pass the empty kv cache inputs
            # (batch_size=0) to the engine. Therefore,
            # we skip the validation
            out = self.engine._eng_net.execute_list_out(
                inputs, kv_cache.engine_internal_cache
            )
        else:
            # run the engine without the LIB.kv_cache object
            out = (
                super()
                .run(EngineOperatorInputs(engine_inputs=inputs), **kwargs)
                .get("engine_outputs")
            )

        logits, *kv_cache_state = out
        self._update_kv_cache(
            kv_cache_state=kv_cache_state,
            input_ids_len=self.input_ids_length,
            kv_cache=kv_cache,
        )

        output = {
            "logits": logits,
            "kv_cache": kv_cache,
            "tokens": inp.tokens,
            "in_generation": inp.in_generation,
        }
        return output

    def _add_kv_cache_to_input(self, engine_input, kv_cache):
        kv_cache_state = copy.copy(kv_cache.cached_inputs)

        for idx, input_name in enumerate(self.onnx_input_names_no_cache):
            kv_cache_state[input_name] = engine_input[idx]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def _update_kv_cache(self, kv_cache_state, input_ids_len, kv_cache):
        if bool(kv_cache.engine_internal_cache):
            kv_cache.total_num_processed_tokens += input_ids_len
            return

        kv_cache_state = {
            name: array
            for name, array in zip(self.onnx_input_names_cached, kv_cache_state)
        }

        kv_cache.update(
            state=kv_cache_state,
            input_ids_len=input_ids_len,
        )

    @property
    def onnx_input_names_no_cache(self) -> List[str]:
        """
        :return: The input names for the onnx model, excluding
            the potential kv cache inputs
        """
        return [
            name
            for name in self.engine.input_names
            if not name.startswith(CACHE_INPUT_PREFIX)
        ]

    @property
    def onnx_input_names_cached(self) -> List[str]:
        """
        :return: The cached input names for the onnx model
        """
        return [
            name
            for name in self.engine.input_names
            if name.startswith(CACHE_INPUT_PREFIX)
        ]

    @property
    def cache_shape(self) -> Tuple[int, int, int, int]:
        """
        :return: The shape of the kv cache inputs
            for the onnx model. The shape is
            (batch_size, num_heads, sequence_length, hidden_size)
        """
        cache_engine_input_index = next(
            i
            for i, name in enumerate(self.engine.input_names)
            if CACHE_INPUT_PREFIX in name
        )
        return self.engine.input_shapes[cache_engine_input_index]

    @property
    def output_names(self) -> List[str]:
        """
        :return: The output names for the onnx model
        """
        return self.engine.output_names


class NlEngineOperatorNoCache(EngineOperator):

    input_schema = NlEngineInputNoCache
    output_schema = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, inp: NlEngineInputNoCache, **kwargs) -> Any:
        engine_inputs = [inp.input_ids, inp.attention_mask]
        logits = (
            super()
            .run(EngineOperatorInputs(engine_inputs=engine_inputs), **kwargs)
            .get("engine_outputs")
        )

        logits = numpy.compress(inp.attention_mask[0], logits[0], axis=1)
        return {
            "logits": [logits],
            "logits_shape": None,
            "deterministic": None,
            "kv_cache": None,
            "tokens": None,
            "sampling_temperature": None,
        }, {"prompt_logits": [logits]}
