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
from transformers import AutoTokenizer

from deepsparse.engine import Context
from deepsparse.pipeline import DEEPSPARSE_ENGINE, create_engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.helpers import generate_session_id
from deepsparse.transformers.utils.timings import TextGenerationTimings
from deepsparse.utils import TimerManager
from deepsparse.transformers.utils.helpers import (
    overwrite_onnx_model_inputs_for_kv_cache_models,
)
from deepsparse.transformers.utils.storage_kv_cache import SessionStorageKVCache
from deepsparse.utils.data import numpy_softmax
from deepsparse.utils.onnx import (
    CACHE_INPUT_PREFIX,
    CACHE_OUTPUT_PREFIX,
    overwrite_onnx_model_inputs_for_kv_cache_models,
)


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine"]


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
        tokenizer: AutoTokenizer,
        sampling_temperature: float = 1.0,
        deterministic: bool = True,
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

        kv_cache_enabled = False
        if sum(output_indices_to_be_cached):
            kv_cache_enabled = True
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
        self.sampling_temperature = sampling_temperature
        self.deterministic = deterministic
        self.input_ids_length = input_ids_length
        self.cache_length = sequence_length - input_ids_length
        self.kv_cache_enabled = kv_cache_enabled
        self.kv_cache_storage = SessionStorageKVCache() if kv_cache_enabled else None
        self._freeze_first_position = self._should_freeze_first_position(tokenizer)
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

    def total_num_processed_tokens(self, session_id: str) -> int:
        """
        Fetch the total number of processed tokens for the kv
        session related to session_id identifier.

        :param session_id: The identifier for session to fetch
            the total number of processed tokens for
        :return: The total number of processed tokens for the
            kv session
        """
        session = self.kv_cache_storage.get(session_id)
        if session is None:
            session = self.initialize_session(session_id)
        return session.total_num_processed_tokens

    @property
    def internal_cache_active(self) -> bool:
        """
        :return: Whether the internal kv cache is active
        """
        return self.kv_cache_enabled and self.kv_cache_storage.internal_cache_active

    def run(
        self, inputs: List[numpy.ndarray], session_id: str, val_inp: bool
    ) -> List[numpy.ndarray]:
        """
        Run the engine with the given inputs.

        If the self.internal_cache_active=True, the internal
        deepsparse kv cache management is enabled. In this case
        the LIB.kv_cache class object will be passed to the engine
        call as well.

        :param inputs: The inputs to run the engine with
        :param session_id: The session id to potentially fetch
            the engine internal cache for
        :param val_inp: Whether the input is for validation or not
        :return: The output of the engine
        """

        if self.internal_cache_active:
            # validate the inputs if needed
            if val_inp:
                self.engine._validate_inputs(inputs)
            # run the engine with the LIB.kv_cache object
            engine_internal_cache = self.kv_cache_storage.get(
                session_id
            ).engine_internal_cache
            return self.engine._eng_net.execute_list_out(inputs, engine_internal_cache)
        # run the engine without the LIB.kv_cache object
        return self.engine.run(inputs, val_inp)

    def __call__(
        self,
        inp: List[numpy.ndarray],
        session_id: str,
        val_inp: bool = True,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        The main entry point for running the engine.

        :param inp: The input to run the engine with. We expect a
            list of numpy arrays that contain the input ids,
            attention mask, and position ids (optionally)
        :param session_id: The session id to run the engine with
        :param val_inp: Whether the input is for validation or not
        :return: The generated token and corresponding logits
        """
        timer = self.timer_manager.current
        if self.kv_cache_enabled:
            # if model has kv cache enabled, we need
            # to add the kv cache state to the input
            inp = self.add_kv_cache_to_input(inp, session_id)

        with timer.time(f"EXECUTE_ENGINE_SEQ_LEN_{self.sequence_length}"):
            out = self.run(inp, session_id, val_inp)

        if self.kv_cache_enabled:
            with timer.time(TextGenerationTimings.KV_CACHE_UPDATE):
                logits, *kv_cache_state = out
                self.update_kv_cache(
                    kv_cache_state=kv_cache_state,
                    input_ids_len=self.input_ids_length,
                    session_id=session_id,
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

    def initialize_session(self, session_id: str) -> DecoderKVCache:
        """
        Initializes the session for the engine and puts it into storage.
        Note, if the session with the `session_id` is already in the storage,
        this method will overwrite it.

        :param session_id: The session id to initialize the session for
        :return: The initialized session
        """
        kv_cache_state = self._initialize_kv_cache_state(self.capacity)
        session = DecoderKVCache(use_deepsparse_cache=self.use_deepsparse_cache)
        session.setup(
            session_id=session_id,
            state=kv_cache_state,
            num_processed_tokens=0,
            freeze_first_position=self._freeze_first_position,
        )

        self.kv_cache_storage.put(session)
        return session

    def transfer_cache_session(self, session: DecoderKVCache):
        """
        Update the kv cache storage of this decoder engine with
        an external DecoderKVCache object.

        :param session: The external DecoderKVCache object
        """
        self.kv_cache_storage.put(session)

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

    def add_kv_cache_to_input(
        self, inp: List[numpy.ndarray], session_id: str
    ) -> List[numpy.ndarray]:
        """
        Takes the input and adds the past kv cache state to it.

        If the internal kv cache is active, the kv cache state
        will always be reinitialized to zeros. This is just to make sure
        that the input shapes of the kv cache arrays to the
        model are correct, the actual values are
        being tracked internally inside the engine.

        If the internal kv cache is not active, we need to
        fetch the kv cache state as numpy arrays
        from the current kv cache storage. If no session
        present, initialize it if required.

        :param inp: The input to the model
        :param session_id: The session id to fetch the kv cache state for
        :return The input with the kv cache state added to it
        """
        if self.internal_cache_active:
            kv_cache_state = self._initialize_kv_cache_state(self.capacity)
        else:
            session = self.kv_cache_storage.get(session_id)
            if session is None:
                session = self.initialize_session(session_id)
            else:
                session.set_capacity(self.capacity)
            kv_cache_state = session.cached_inputs

        for idx, input_name in enumerate(self.onnx_input_names_no_cache):
            kv_cache_state[input_name] = inp[idx]

        new_inp = [kv_cache_state[name] for name in self.engine.input_names]
        return new_inp

    def update_kv_cache(
        self,
        kv_cache_state: List[numpy.ndarray],
        input_ids_len: int,
        session_id: str,
    ):
        """
        Pull the appropriate session from the kv cache storage
        and update the kv cache state with the new kv cache state.

        If the internal kv cache is enabled, we refrain from
        updating the kv cache state as it is being tracked internally
        inside the engine. We only update the number of tokens processed.

        :param kv_cache_state: The state of the kv cache storage
        :param input_ids_len: The length of input_ids
        :param session_id: The session id to fetch the kv cache session for
        """
        if self.internal_cache_active:
            session = self.kv_cache_storage.get(session_id)
            session.total_num_processed_tokens += input_ids_len
            return

        cache_onnx_names = [
            name
            for name in self.engine.input_names
            if name.startswith(CACHE_INPUT_PREFIX)
        ]
        kv_cache_state = {
            name: array for name, array in zip(cache_onnx_names, kv_cache_state)
        }

        session = self.kv_cache_storage.get(session_id)
        session.update(state=kv_cache_state, input_ids_len=input_ids_len)
        self.kv_cache_storage.put(session)

    def _initialize_kv_cache_state(self, length: int) -> Dict[str, numpy.ndarray]:
        # initialize empty kv cache of size
        # (batch_size, num_attention_heads, length, hidden_dims)

        cache_engine_input_index = next(
            i
            for i, name in enumerate(self.engine.input_names)
            if CACHE_INPUT_PREFIX in name
        )
        batch_size, num_attention_heads, _, hidden_dims = self.engine.input_shapes[
            cache_engine_input_index
        ]

        empty_kv_cache_tensor = numpy.zeros(
            (batch_size, num_attention_heads, length, hidden_dims),
            dtype=self.kv_cache_data_type,
        )

        cache_keys = [
            output_name.replace(CACHE_OUTPUT_PREFIX, CACHE_INPUT_PREFIX)
            for output_name in self.engine.output_names
            if output_name.startswith(CACHE_OUTPUT_PREFIX)
        ]
        return {key: empty_kv_cache_tensor for key in cache_keys}

    @staticmethod
    def _should_freeze_first_position(tokenizer) -> bool:
        # use tokenizer to find out whether we should freeze the first position
        # (True if tokenizer has a prefix for a BOS token)
        if tokenizer is None:
            return False
        if hasattr(tokenizer, "add_bos_token"):
            return True
        return False
