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
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy

from deepsparse.engine import Context
from deepsparse.pipeline import DEEPSPARSE_ENGINE, create_engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.timings import TextGenerationTimings
from deepsparse.utils import TimerManager
from deepsparse.utils.onnx import (
    CACHE_INPUT_PREFIX,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine"]


class NLDecoderEngine:
    """
    The NLDecoderEngine (Natural Language Decoder Engine) handles the
    logic around the inference for Natural Language pipeline,
    including batching and kv cache manipulation logic.

    :param onnx_file_path: The path to the onnx model file
    :param engine_type: The type of engine to use for the inference
    :param engine_args: The arguments to pass to the engine
    :param sequence_length: The maximum sequence length to run the engine for
    :param input_ids_length: The maximum input ids length to run the engine for
    :param engine_context: The context to run the engine in
    :param internal_kv_cache: Whether to use the deepsparse
        kv cache in the DecoderKVCache object or not
    """

    def __init__(
        self,
        onnx_file_path: str,
        engine_type: str,
        engine_args: Dict[str, Any],
        sequence_length: int,
        input_ids_length: int,
        engine_context: Optional[Context] = None,
        internal_kv_cache=False,
        timer_manager: TimerManager = None,
    ):
        # flag to indicate if the model is quantized or not
        self.kv_cache_data_type = None
        (
            onnx_file_path,
            output_indices_to_be_cached,
            kv_cache_data_type,
        ) = overwrite_onnx_model_inputs_for_kv_cache_models(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )

        if any(output_indices_to_be_cached):
            self.kv_cache_data_type = kv_cache_data_type
            if internal_kv_cache and engine_type == DEEPSPARSE_ENGINE:
                # inform the engine, that are using the kv cache
                engine_args["cached_outputs"] = output_indices_to_be_cached

        self.engine = create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=engine_type,
            engine_args=engine_args,
            context=engine_context,
        )
        self.timer_manager = timer_manager or TimerManager()
        self.sequence_length = sequence_length
        self.input_ids_length = input_ids_length
        self.cache_length = sequence_length - input_ids_length
        self._engine_type = engine_type

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

    def run(
        self, inputs: List[numpy.ndarray], val_inp: bool, kv_cache: DecoderKVCache
    ) -> List[numpy.ndarray]:
        """
        Run the engine with the given inputs.
        If the kv_cache.engine_internal_cache=True, the internal
        deepsparse kv cache management is enabled. In this case
        the LIB.kv_cache class object will be passed to the engine
        call as well. In this scenario also the inputs will not be
        validated, even if the val_inp=True. This is because we
        want to pass the empty kv cache inputs (batch_size=0) to
        the engine.

        :param inputs: The inputs to run the engine with
        :param val_inp: Whether the input is for validation or not
        :param kv_cache: The kv cache object to use for the inference

        :return: The output of the engine
        """
        if kv_cache is None:
            # run the engine without the kv cache support
            return self.engine.run(inputs, val_inp)

        if bool(kv_cache.engine_internal_cache):
            # run the engine assuming internal kv cache
            # management. In this case the LIB.kv_cache
            # class object will be passed to the engine
            # call as well
            # conventionally, before dispatching
            # inputs to the engine, we validate them
            # if val_inp=True. However, in this case
            # we want to pass the empty kv cache inputs
            # (batch_size=0) to the engine. Therefore,
            # we skip the validation
            return self.engine._eng_net.execute_list_out(
                inputs, kv_cache.engine_internal_cache
            )
        else:
            # run the engine assuming external kv cache
            # management.
            return self.engine.run(inputs, val_inp)

    def __call__(
        self,
        inp: List[numpy.ndarray],
        kv_cache: Optional[DecoderKVCache] = None,
        val_inp: bool = True,
    ) -> numpy.ndarray:
        """
        The main entry point for running the engine.

        :param inp: The input to run the engine with. We expect a
            list of numpy arrays that contain the input ids,
            attention mask, and position ids (optionally)
        :param kv_cache: The DecoderKVCache object that contains
            the kv cache state
        :param val_inp: Whether the input is for validation or not

        :return: The generated token and corresponding logits
        """
        timer = self.timer_manager.current_or_new()
        if kv_cache:
            # if model has kv cache enabled, we need
            # to add the kv cache state to the input
            inp = self.add_kv_cache_to_input(inp, kv_cache)

        with timer.time(f"EXECUTE_ENGINE_SEQ_LEN_{self.sequence_length}"):
            out = self.run(inp, val_inp, kv_cache)

        if kv_cache:
            with timer.time(TextGenerationTimings.KV_CACHE_UPDATE):
                logits, *kv_cache_state = out
                self.update_kv_cache(
                    kv_cache_state=kv_cache_state,
                    input_ids_len=self.input_ids_length,
                    kv_cache=kv_cache,
                )
        else:
            logits = out[0]

        return logits

    def __str__(self):
        return f"{self.__class__.__name__}: {self.engine}"

    def __repr__(self):
        return str(self)

    def add_kv_cache_to_input(
        self, inp: List[numpy.ndarray], kv_cache: DecoderKVCache
    ) -> List[numpy.ndarray]:
        """
        Takes the input and adds the kv cache state to it.

        If the internal kv cache is enabled, the kv cache state
        will always be an empty array. This is just to make sure
        that the input shapes of the kv cache arrays to the
        model are correct, the actual values are
        being tracked internally inside the engine.

        If the internal kv cache is disabled, we need to
        fetch the kv cache state as numpy arrays
        from the current session, or initialize it if required.


        :param inp: The input to the model
        :param kv_cache: The kv cache object

        :return The input with the kv cache state added to it
        """
        kv_cache_state = copy.copy(kv_cache.cached_inputs)

        for idx, input_name in enumerate(self.onnx_input_names_no_cache):
            kv_cache_state[input_name] = inp[idx]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def update_kv_cache(
        self,
        kv_cache_state: List[numpy.ndarray],
        input_ids_len: int,
        kv_cache: DecoderKVCache,
    ):
        """
        Updates the kv cache using the new kv cache state.

        If the internal kv cache is enabled, we refrain from
        updating the kv cache state as it is being tracked internally
        inside the engine. We only update the number of tokens processed.

        :param kv_cache_state: The new state of the kv cache storage
        :param input_ids_len: The length of input_ids
        :param kv_cache: The kv cache object to update
        """
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
