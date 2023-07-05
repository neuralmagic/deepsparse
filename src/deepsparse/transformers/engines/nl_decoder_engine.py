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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy
import onnx
from transformers import AutoTokenizer

from deepsparse.engine import Context
from deepsparse.pipeline import create_engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.helpers import generate_session_id, softmax
from deepsparse.transformers.utils.storage_kv_cache import KVCacheSessionStorage
from sparsezoo.utils.onnx import save_onnx


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine", "synchronise_engines_cache"]

_CACHE_INPUT_NAME = "past_key_values"


def synchronise_engines_cache(engines: Set["NLDecoderEngine"]):  # noqa F821
    """
    Takes a set of engines and synchronises the kv cache storage
    across all of them. This means that the latest kv cache storage
    from all engines will be transferred to the remaining engines.

    :param engines: A set of engines to synchronise the kv cache storage for
    """
    newest_timestamp = None
    recently_updated_engine = None

    for engine in engines:
        if engine.kv_cache_storage is not None:
            timestamp = engine.kv_cache_storage.latest_update_timestamp
            newest_timestamp = (
                timestamp if newest_timestamp is None else newest_timestamp
            )
            if timestamp >= newest_timestamp:
                newest_timestamp = timestamp
                recently_updated_engine = engine

    engines.remove(recently_updated_engine)

    [
        engine.transfer_cache_storage(cache_storage=engine.kv_cache_storage)
        for engine in engines
    ]


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

        onnx_file_path, input_indices_to_be_cached = self.overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            input_ids_length=input_ids_length,
        )
        kv_cache_enabled = False
        if input_indices_to_be_cached:
            # inform the engine, that are using the kv cache
            engine_args["cache_input_bools"] = input_indices_to_be_cached
            kv_cache_enabled = True

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
        self.use_deepsparse_cache = use_deepsparse_cache
        self.kv_cache_storage = KVCacheSessionStorage() if kv_cache_enabled else None

        self._freeze_first_position = self._should_freeze_first_position(tokenizer)

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

    def __str__(self):
        return f"{self.__class__.__name__}: {self.engine}"

    def __call__(
        self,
        inp: List[numpy.ndarray],
        session_id: Optional[str] = None,
        val_inp: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        The main entry point for running the engine.

        :param inp: The input to run the engine with. We expect a
            list of numpy arrays that contain the input ids,
            attention mask, and position ids (optionally)
        :param session_id: The session id to use for the kv cache.
               If None, and the model uses cache, a new UUID for
               session id will be generated
        :param val_inp: Whether the input is for validation or not
        :return: The generated token and corresponding logits
        """
        if self.kv_cache_enabled:
            session_id = session_id or generate_session_id()
            # if kv cache is enabled, we need to add the kv cache state
            # to the input
            inp = self.add_kv_cache_to_input(inp=inp, session_id=session_id)

        out = self.engine.run(inp, val_inp)

        if self.kv_cache_enabled:
            logits, *kv_cache_state = out
            self.update_kv_cache(
                kv_cache_state=kv_cache_state,
                input_ids_len=self.input_ids_length,
                session_id=session_id,
            )
        else:
            logits = out[0]

        token = self.generate_token(logits=logits[:, -1, :])

        return token, logits

    def transfer_cache_storage(self, cache_storage: KVCacheSessionStorage):
        """
        Transfers the kv cache storage to the engine. Call this method when
        you want to transfer the kv cache storage from e.g. one engine to another.

        :param cache_storage: The cache storage to transfer the state from
        """
        _LOGGER.debug(
            f"Transferring cache storage {cache_storage} to the engine {self.engine}"
        )

        self.kv_cache_storage = cache_storage

    @staticmethod
    def overwrite_onnx_model_inputs(
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

        input_indices_to_be_cached = [
            i
            for i, inp in enumerate(model.graph.input)
            if inp.name.startswith(_CACHE_INPUT_NAME)
        ]
        return onnx_file_path, input_indices_to_be_cached

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

    def add_kv_cache_to_input(
        self, inp: List[numpy.ndarray], session_id: str
    ) -> List[numpy.ndarray]:
        """
        Takes the input and adds the past kv cache state to it.
        The state is retrieved from the kv cache storage using the session id.

        :param inp: The input to the model
        :param session_id: The session id to use for
            retrieving the kv cache state from the
            kv cache storage
        :return The input with the kv cache state added to it
        """
        session = self.kv_cache_storage.get(session_id)
        if session is None:
            kv_cache_state = self._initialize_kv_cache_state(
                self.sequence_length - self.input_ids_length
            )
            session = DecoderKVCache(use_deepsparse_cache=self.use_deepsparse_cache)
            session.setup(
                identifier=session_id,
                state=kv_cache_state,
                num_processed_tokens=0,
                freeze_first_position=self._freeze_first_position,
            )
            self.kv_cache_storage.put(session=session)
        else:
            kv_cache_state = session.cached_inputs

        kv_cache_state["input_ids"] = inp[0]
        kv_cache_state["attention_mask"] = inp[1]
        if len(inp) == 3:
            kv_cache_state["positions"] = inp[2]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def update_kv_cache(
        self,
        kv_cache_state: List[numpy.ndarray],
        session_id: str,
        input_ids_len: int,
    ):
        """
        Updates the state of the kv cache storage with the new
        kv cache state corresponding to the session id. The new
        state (output from the inference) is transformed by the
        KVCacheDecoder (`session.update(...)`) so that it can
        be used as an input to the next inference.
        Once the state is transformed, it is  stored in the kv
        cache storage.

        :param kv_cache_state: The state of the kv cache storage
        :param session_id: The session id under which the new
            cache state is stored.
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

        session = self.kv_cache_storage.get(session_id)
        session.update(
            state=kv_cache_state,
            input_ids_len=input_ids_len,
        )
        self.kv_cache_storage.put(session=session)

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

    @staticmethod
    def _should_freeze_first_position(tokenizer) -> bool:
        # use tokenizer to find out whether we should freeze the first position
        # (True if tokenizer has a prefix for a BOS token)
        if tokenizer is None:
            return False
        if hasattr(tokenizer, "bos_token"):
            return True
        return False
