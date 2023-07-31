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
import onnx
from transformers import AutoTokenizer

from deepsparse.engine import Context
from deepsparse.pipeline import DEEPSPARSE_ENGINE, create_engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.helpers import generate_session_id
from deepsparse.utils.data import numpy_softmax
from deepsparse.utils.onnx import translate_onnx_type_to_numpy
from sparsezoo.utils.onnx import save_onnx


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine"]

_CACHE_INPUT_NAME = "past_key_values"


class NLDecoderEngine:
    """
    The NLDecoderEngine (NaturalLanguageDecoderEngine) handles the
    logic around the inference for Natural Language pipeline,
    including batching and kv cache logic.

    :param onnx_file_path: The path to the onnx model file
    :param engine_type: The type of engine to use for the inference
    :param engine_args: The arguments to pass to the engine
    :param sequence_length: The maximum sequence length to run the engine for
    :param input_ids_length: The maximum input ids length to run the engine for
    :param engine_context: The context to run the engine in
    :param sampling_temperature: The temperature to use for sampling
    :param deterministic: Whether to use deterministic sampling
    :param tokenizer: The tokenizer to used for engine inputs
    :param engine_context: The context to run the engine in
    :param use_deepsparse_cache: Whether to use the deepsparse
        kv cache in the DecoderKVCache object or not
    """

    def __init__(
        self,
        onnx_file_path: str,
        engine_type: str,
        engine_args: Dict[str, Any],
        sequence_length: int,
        input_ids_length: int,
        tokenizer: AutoTokenizer,
        sampling_temperature: float = 1.0,
        deterministic: bool = True,
        engine_context: Optional[Context] = None,
        use_deepsparse_cache=False,
    ):
        # flag to indicate if the model is quantized or not
        self.kv_cache_data_type = None

        onnx_file_path, output_indices_to_be_cached = self.overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )
        kv_cache_enabled = False
        if sum(output_indices_to_be_cached):
            kv_cache_enabled = True
            if use_deepsparse_cache and engine_type == DEEPSPARSE_ENGINE:
                # inform the engine, that are using the kv cache
                engine_args["cache_output_bools"] = output_indices_to_be_cached

        self.engine = create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=engine_type,
            engine_args=engine_args,
            context=engine_context,
        )
        self.sequence_length = sequence_length
        self.sampling_temperature = sampling_temperature
        self.deterministic = deterministic
        self.input_ids_length = input_ids_length
        self.kv_cache_enabled = kv_cache_enabled
        self.kv_cache = (
            DecoderKVCache(use_deepsparse_cache) if kv_cache_enabled else None
        )
        self._freeze_first_position = self._should_freeze_first_position(tokenizer)
        self._session_id = generate_session_id()

    @property
    def session_id(self) -> str:
        """
        :return: The session id for the kv_cache if enabled
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str):
        """
        :param session_id: The session id to set for the kv_cache
        """
        self._session_id = session_id

    @property
    def onnx_input_names_no_cache(self) -> List[str]:
        """
        :return: The input names for the onnx model, excluding
            the potential kv cache inputs
        """
        return [
            name
            for name in self.engine.input_names
            if not name.startswith(_CACHE_INPUT_NAME)
        ]

    @property
    def num_non_blank_cache_entries(self) -> int:
        """
        :return a number of non-blank entries in the
        kv cache
        """
        return self.kv_cache.num_non_blank_entries

    def __call__(
        self,
        inp: List[numpy.ndarray],
        val_inp: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        The main entry point for running the engine.

        :param inp: The input to run the engine with. We expect a
            list of numpy arrays that contain the input ids,
            attention mask, and position ids (optionally)
        :param val_inp: Whether the input is for validation or not
        :return: The generated token and corresponding logits
        """
        if self.kv_cache:
            # if kv cache is enabled, we need to add the kv cache state
            # to the input
            inp = self.add_kv_cache_to_input(inp)

        out = self.engine.run(inp, val_inp)

        if self.kv_cache:
            logits, *kv_cache_state = out
            self.update_kv_cache(
                kv_cache_state=kv_cache_state, input_ids_len=self.input_ids_length
            )
        else:
            logits = out[0]

        # select batch idx 0, batch is always 1
        token = self.generate_token(logits=logits[0, -1, :])

        return token, logits

    def __str__(self):
        return f"{self.__class__.__name__}: {self.engine}"

    def __repr__(self):
        return str(self)

    def transfer_cache_state(self, cache: DecoderKVCache):
        """
        Transfers the kv cache state and the number of tokens processed
        information from another NLDecoderEngine. Call this method when
        you want to transfer the kv cache state from one engine to another.

        This method will also automatically set the kv cache capacity to
        the appropriate value for the new engine.

        :param cache: The `DecoderKVCache` object to transfer to the engine
            from
        """
        cache_to_copy = copy.deepcopy(cache)
        target_cache_capacity = self.sequence_length - self.input_ids_length
        cache_to_copy.set_capacity(target_cache_capacity)
        self.kv_cache = cache_to_copy

    def overwrite_onnx_model_inputs(
        self,
        onnx_file_path: str,
        sequence_length: int,
        input_ids_length: int,
        batch_size: int = 1,
    ) -> Tuple[str, List[int]]:
        """
        Enforces the appropriate input shapes for the onnx model, as well as
        checks whether kv cache is enabled or not.

        :param onnx_file_path: The path to the onnx model file that will be
            overwritten with the new input shapes
        :param batch_size: The batch size to use for the input
        :param sequence_length: The sequence length to use for the input
        :param input_ids_length: The length of input_ids
        :return: The path to the onnx model file that has been overwritten
            with the new input shapes, as well as the indices of the inputs
            that should be cached
        """
        model = onnx.load(onnx_file_path, load_external_data=False)
        initializer_input_names = set(node.name for node in model.graph.initializer)
        external_inputs = [
            inp for inp in model.graph.input if inp.name not in initializer_input_names
        ]
        for external_input in external_inputs:
            # overwrite the batch size for all the inputs
            external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

            if external_input.name in ["input_ids", "positions"]:
                external_input.type.tensor_type.shape.dim[
                    1
                ].dim_value = input_ids_length
            elif external_input.name == "attention_mask":
                external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length
            elif external_input.name.startswith(_CACHE_INPUT_NAME):
                external_input.type.tensor_type.shape.dim[2].dim_value = (
                    sequence_length - input_ids_length
                )
            elif external_input.name.startswith("causal_mask"):
                external_input.type.tensor_type.shape.dim[
                    2
                ].dim_value = input_ids_length
                external_input.type.tensor_type.shape.dim[3].dim_value = sequence_length
            else:
                raise ValueError(
                    f"Unexpected external input name: {external_input.name}"
                )

        _LOGGER.info(
            "Overwriting in-place the input shapes "
            f"of the transformer model at {onnx_file_path}"
        )
        save_onnx(model, onnx_file_path)

        output_indices_to_be_cached = [
            1 if inp.name.startswith("present") else 0 for inp in model.graph.output
        ]
        if any(output_indices_to_be_cached):
            kv_cache_elem_type = next(
                inp
                for inp in model.graph.input
                if inp.name.startswith(_CACHE_INPUT_NAME)
            ).type.tensor_type.elem_type
            self.kv_cache_data_type = translate_onnx_type_to_numpy(kv_cache_elem_type)

        return onnx_file_path, output_indices_to_be_cached

    def generate_token(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits using the sampling temperature.

        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """
        if self.deterministic:
            return numpy.argmax(logits)

        logits /= self.sampling_temperature

        probs = numpy_softmax(logits)

        return numpy.random.choice(len(probs), p=probs)

    def reset_kv_cache(self):
        """
        Resets the kv cache state.
        """
        kv_cache_state = self._initialize_kv_cache_state(
            self.sequence_length - self.input_ids_length
        )
        self.kv_cache.setup_session(
            session_id=self._session_id,
            state=kv_cache_state,
            num_processed_tokens=0,
            freeze_first_position=self._freeze_first_position,
        )

    def add_kv_cache_to_input(self, inp: List[numpy.ndarray]) -> List[numpy.ndarray]:
        """
        Takes the input and adds the past kv cache state to it.

        :param inp: The input to the model
        :return The input with the kv cache state added to it
        """
        kv_cache_state = self.kv_cache.cached_inputs
        if kv_cache_state is None:
            self.reset_kv_cache()
            kv_cache_state = self.kv_cache.cached_inputs

        for idx, input_name in enumerate(self.onnx_input_names_no_cache):
            kv_cache_state[input_name] = inp[idx]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def update_kv_cache(
        self,
        kv_cache_state: List[numpy.ndarray],
        input_ids_len: int,
    ):
        """
        Updates the state of the kv cache

        :param kv_cache_state: The state of the kv cache storage
        :param input_ids_len: The length of input_ids
        """
        cache_onnx_names = [
            name
            for name in self.engine.input_names
            if name.startswith(_CACHE_INPUT_NAME)
        ]
        kv_cache_state = {
            name: array for name, array in zip(cache_onnx_names, kv_cache_state)
        }

        self.kv_cache.update_session(
            state=kv_cache_state,
            input_ids_len=input_ids_len,
        )

    def _initialize_kv_cache_state(self, length: int) -> Dict[str, numpy.ndarray]:
        # initialize empty kv cache of size
        # (batch_size, num_attention_heads, length, hidden_dims)

        cache_engine_input_index = next(
            i
            for i, name in enumerate(self.engine.input_names)
            if _CACHE_INPUT_NAME in name
        )
        batch_size, num_attention_heads, _, hidden_dims = self.engine.input_shapes[
            cache_engine_input_index
        ]

        empty_kv_cache_tensor = numpy.zeros(
            (batch_size, num_attention_heads, length, hidden_dims),
            dtype=self.kv_cache_data_type,
        )

        cache_keys = [
            output_name.replace("present", _CACHE_INPUT_NAME)
            for output_name in self.engine.output_names
            if output_name.startswith("present")
        ]
        return {key: empty_kv_cache_tensor for key in cache_keys}

    @staticmethod
    def _should_freeze_first_position(tokenizer) -> bool:
        # use tokenizer to find out whether we should freeze the first position
        # (True if tokenizer has a prefix for a BOS token)
        if tokenizer is None:
            return False
        if hasattr(tokenizer, "bos_token"):
            return True
        return False
