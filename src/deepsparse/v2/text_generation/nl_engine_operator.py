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
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy
from pydantic import BaseModel, Field

from deepsparse.utils import join_engine_outputs, split_engine_inputs
from deepsparse.utils.onnx import (
    CACHE_INPUT_PREFIX,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)
from deepsparse.v2.operators.engine_operator import (
    DEEPSPARSE_ENGINE,
    EngineOperator,
    EngineOperatorInputs,
)


__all__ = ["NLEngineOperator", "NLEngineInputs"]


class NLEngineInputs(BaseModel):
    engine_inputs: List = Field(description="engine_inputs")
    kv_cache: Any = Field(description="kv_cache object")
    tokens: List = Field(description="tokens")
    in_generation: Any = Field(description="in_generation", default=None)
    engine: Optional[Any] = Field(
        description="override the engine to run forward pass with",
        default=None,
    )

    @classmethod
    def join(cls, inputs: List["NLEngineInputs"]) -> "NLEngineInputs":
        """
        :param inputs: list of separate EngineOperatorInputs, batch size must be 1
        :return: list of inputs joined into a single input with a multi batch size
        """
        all_engine_inputs = []
        all_kv_cache = []
        all_tokens = []
        all_generation = []

        for engine_input in inputs:
            all_engine_inputs.append(engine_input.engine_inputs)
            all_kv_cache.append(engine_input.kv_cache)
            all_tokens.append(engine_input.tokens)
            all_generation.append(engine_input.in_generation)

        for engine_inputs in all_engine_inputs:
            if engine_inputs[0].shape[0] != 1:
                raise RuntimeError(
                    "join requires all inputs to have batch size 1, found input with "
                    f"batch size {engine_inputs[0].shape[0]}"
                )
        return cls(
            engine_inputs=all_engine_inputs,
            tokens=all_tokens,
            in_generation=all_generation,
            kv_cache=all_kv_cache,
        )

    class Config:
        arbitrary_types_allowed = True


class NLEngineOutputs(BaseModel):
    engine_outputs: Any = Field(description="engine_outputs")
    kv_cache: Any = Field(description="kv_cache object")
    tokens: List = Field(description="tokens")
    in_generation: Any = Field(description="in_generation", default=None)

    def split(self) -> List["NLEngineOutputs"]:
        """
        :return: list of the current outputs split to a batch size of 1 each
        """
        split_outputs = [
            numpy.expand_dims(self.engine_outputs[i], 0)
            for i in range(len(self.engine_outputs))
        ]
        return [
            self.__class__(
                engine_outputs=split_outputs[i],
                kv_cache=self.kv_cache[i],
                tokens=self.tokens[i],
                in_generation=self.in_generation[i],
            )
            for i in range(len(split_outputs))
        ]


class NLEngineOperator(EngineOperator):

    """
    Operator for the NL Decoder Engine. This Operator inherits from the EngineOperator.
    Specific updates to engine attributes are made through this operator, as well
    as updating the kv_cache. This Operator is used for both the single-token and
    multi-token case.
    """

    input_schema = NLEngineInputs
    output_schema = NLEngineOutputs

    def __init__(
        self,
        sequence_length: int,
        input_ids_length: int,
        internal_kv_cache: bool = False,
        **kwargs,
    ):

        self.sequence_length = sequence_length
        self.input_ids_length = input_ids_length
        self.kv_cache_data_type = None
        self.internal_kv_cache = internal_kv_cache
        self.model_path = kwargs.get("model_path")
        (onnx_file_path, additional_outputs) = self.override_model_inputs(
            self.model_path, batch_size=1, return_additional_outputs=True
        )
        output_indices_to_be_cached, kv_cache_data_type, = additional_outputs.get(
            "output_indices_to_be_cached"
        ), additional_outputs.get("kv_cache_data_type")

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

    def override_model_inputs(
        self,
        model_path: Union[str, Path],
        batch_size: int,
        return_additional_outputs=False,
    ):
        """
        Override the model based on the provided batch_size, sequence_length,
        and input_ids_length.

        :param model_path: Path to the model
        :param batch_size: The batch size to be used for the model
        :return: new overwritten model file path. Optionally returns additional outputs
            specific to the NLDecoder engine
        """
        (
            onnx_file_path,
            output_indices_to_be_cached,
            kv_cache_data_type,
        ) = overwrite_onnx_model_inputs_for_kv_cache_models(
            onnx_file_path=model_path,
            batch_size=batch_size,
            sequence_length=self.sequence_length,
            input_ids_length=self.input_ids_length,
        )
        if return_additional_outputs:
            return onnx_file_path, {
                "output_indices_to_be_cached": output_indices_to_be_cached,
                "kv_cache_data_type": kv_cache_data_type,
            }
        return onnx_file_path

    def run(self, inp: NLEngineInputs, **kwargs) -> NLEngineOutputs:
        engine_input = inp.engine_inputs
        kv_cache = inp.kv_cache

        split = True
        if not isinstance(kv_cache, list):
            split = False
            kv_cache = [kv_cache]
            engine_input = [engine_input]

        inputs = list(map(self._add_kv_cache_to_input, engine_input, kv_cache))

        if bool(kv_cache[0].engine_internal_cache):
            # conventionally, before dispatching
            # inputs to the engine, we validate them
            # if val_inp=True. However, in this case
            # we want to pass the empty kv cache inputs
            # (batch_size=0) to the engine. Therefore,
            # we skip the validation

            # Internal kv_cache works for batch_size of 1 atm
            out = self.engine._eng_net.execute_list_out(
                inputs[0], kv_cache[0].engine_internal_cache
            )
        else:
            # run the engine without the LIB.kv_cache object
            # stack multiple batch inputs along the batch dimension
            inputs = join_engine_outputs(inputs, len(inputs))
            out = (
                super()
                .run(
                    EngineOperatorInputs(engine_inputs=inputs, engine=inp.engine),
                    **kwargs,
                )
                .get("engine_outputs")
            )

        # logits should be stacked along batch dim
        # kv_cache_state should be a list where each dim 0 is batch_size
        logits, *kv_cache_state = out
        kv_cache_state, _ = split_engine_inputs(kv_cache_state, 1)

        if len(kv_cache_state) > 0:
            for i in range(len(kv_cache)):
                self._update_kv_cache(
                    kv_cache_state=kv_cache_state[i], kv_cache=kv_cache[i]
                )
        else:
            # internal kv cache case
            self._update_kv_cache(kv_cache=kv_cache[0])

        output = {
            "engine_outputs": logits,
            "kv_cache": kv_cache if split else kv_cache[0],
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

    def _update_kv_cache(self, kv_cache, kv_cache_state=None):
        if bool(kv_cache.engine_internal_cache):
            kv_cache.total_num_processed_tokens += self.input_ids_length
            return

        kv_cache_state = {
            name: array
            for name, array in zip(self.onnx_input_names_cached, kv_cache_state)
        }

        kv_cache.update(state=kv_cache_state, input_ids_len=self.input_ids_length)

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
