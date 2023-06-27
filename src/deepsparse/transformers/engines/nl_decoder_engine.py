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
from typing import Any, Dict, List, Optional, Tuple

import numpy
import onnx
from transformers import AutoTokenizer

from deepsparse.engine import Context
from deepsparse.pipeline import create_engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.helpers import generate_session_id, softmax
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

        onnx_file_path, kv_cache_enabled = self.overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )

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
        self._num_tokens = 0  # the number of tokens processed so far
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
            inp = self._add_kv_cache_to_input(inp)

        out = self.engine.run(inp, val_inp)

        self._num_tokens += self.input_ids_length

        if self.kv_cache:
            logits, *kv_cache_state = out
            self._update_kv_cache(
                kv_cache_state=kv_cache_state,
            )
        else:
            logits = out[0]
            B, S, V = logits.shape  # batch, sequence, vocab
            logits = logits[:, -1, :].reshape(B, 1, V)  # only take the last token

        token = self.generate_token(logits=logits)

        return token, logits

    def transfer_cache_state(self, engine: "NLDecoderEngine"):
        """
        Transfers the kv cache state and the number of tokens processed
        information from another NLDecoderEngine. Call this method when
        you want to transfer the kv cache state from one engine to another.

        :param engine: The `NLDecoderEngine` to transfer the kv cache state
            from
        """
        state = engine.kv_cache.cached_inputs

        self.kv_cache.setup_session(
            session_id=self._session_id,
            state=state,
            sequence_length=self.sequence_length,
            freeze_first_position=self._freeze_first_position,
        )
        # maybe set as a property for more control?
        self._num_tokens = engine._num_tokens

    @staticmethod
    def overwrite_onnx_model_inputs(
        onnx_file_path: str,
        sequence_length: int,
        input_ids_length: int,
        batch_size: int = 1,
    ) -> Tuple[str, bool]:
        """
        Enforces the appropriate input shapes for the onnx model, as well as
        checks whether kv cache is enabled or not.

        :param onnx_file_path: The path to the onnx model file that will be
            overwritten with the new input shapes
        :param batch_size: The batch size to use for the input
        :param sequence_length: The sequence length to use for the input
        :param input_ids_length: The length of input_ids
        :return: The path to the onnx model file that has been overwritten
            with the new input shapes, as well as whether kv cache is enabled
            or not
        """
        model = onnx.load(onnx_file_path, load_external_data=False)
        initializer_input_names = set([node.name for node in model.graph.initializer])
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
            else:
                raise ValueError(
                    f"Unexpected external input name: {external_input.name}"
                )

        _LOGGER.info(
            "Overwriting in-place the input shapes "
            f"of the transformer model at {onnx_file_path}"
        )
        save_onnx(model, onnx_file_path)

        is_cache_enabled = any(
            _CACHE_INPUT_NAME in node.name for node in model.graph.input
        )
        return onnx_file_path, is_cache_enabled

    def generate_token(self, logits: numpy.ndarray) -> numpy.ndarray:
        """
        Samples a token from the logits using the sampling temperature.
        :param logits: the logits from the model with shape (vocab_size,)
        :return: the sampled token
        """
        if self.deterministic:
            return numpy.argmax(logits)

        logits /= self.sampling_temperature

        probs = softmax(logits)

        return numpy.random.choice(len(probs), p=probs)

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
            dtype=numpy.float32,
        )

        cache_keys = [
            output_name.replace("present", _CACHE_INPUT_NAME)
            for output_name in self.engine.output_names
            if output_name.startswith("present")
        ]
        return {key: empty_kv_cache_tensor for key in cache_keys}

    def _add_kv_cache_to_input(self, inp: List[numpy.ndarray]) -> List[numpy.ndarray]:
        kv_cache_state = self.kv_cache.cached_inputs
        if kv_cache_state is None:
            kv_cache_state = self._initialize_kv_cache_state(
                self.sequence_length - self.input_ids_length
            )
            self.kv_cache.setup_session(
                session_id=self._session_id,
                state=kv_cache_state,
                sequence_length=self.sequence_length,
                freeze_first_position=self._freeze_first_position,
            )

        kv_cache_state["input_ids"] = inp[0]
        kv_cache_state["attention_mask"] = inp[1]
        if len(inp) == 3:
            kv_cache_state["positions"] = inp[2]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def _update_kv_cache(
        self,
        kv_cache_state: List[numpy.ndarray],
    ):
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
            num_tokens=self._num_tokens,
            # TODO: Make it more general once
            # multitoken regression is supported
            input_ids_len=1,
        )

    @staticmethod
    def _should_freeze_first_position(tokenizer) -> bool:
        # use tokenizer to find out whether we should freeze the first position
        # (True if tokenizer has a prefix for a BOS token)
        if tokenizer is None:
            return False
        if hasattr(tokenizer, "bos_token"):
            return True
        return False
