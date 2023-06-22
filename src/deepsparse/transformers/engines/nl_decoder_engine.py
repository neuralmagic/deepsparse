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

from deepsparse import Pipeline
from deepsparse.engine import Context, Engine
from deepsparse.transformers.utils.decoder_kv_cache import DecoderKVCache
from deepsparse.transformers.utils.helpers import generate_session_id, softmax
from sparsezoo.utils.onnx import save_onnx


_LOGGER = logging.getLogger(__name__)

__all__ = ["NLDecoderEngine"]

_CACHE_INPUT_NAME = "past_key_values"


class NLDecoderEngine(Engine):
    """
    The NLDecoderEngine (NaturalLanguageDecoderEngine) handles the
    logic around the inference for Natural Language pipeline,
    including batching and kv cache logic.

    :param onnx_file_path: The path to the onnx model file
    :param engine_type: The type of engine to use for the inference
    :param multitoken: Whether to run the engine in multitoken mode
    :param engine_args: The arguments to pass to the engine
    :param sequence_length: The maximum sequence length to run the engine for
    :param engine_context: The context to run the engine in
    :param sampling_temperature: The temperature to use for sampling
    :param deterministic: Whether to use deterministic sampling
    """

    def __init__(
        self,
        onnx_file_path: str,
        engine_type: str,
        multitoken: bool,
        engine_args: Dict[str, Any],
        # TODO: having a sequence_length as an argument
        # to the engine is not ideal. The engine should
        # not need to know about it, but it is required
        # for determining whether kv cache is present or not
        sequence_length: int,
        engine_context: Optional[Context] = None,
        sampling_temperature: float = 1.0,
        deterministic: bool = True,
    ):

        onnx_file_path, kv_cache_enabled = self.overwrite_onnx_model_inputs(
            onnx_file_path=onnx_file_path,
            batch_size=engine_args.get("batch_size", 1),
            sequence_length=sequence_length,
            multitoken=multitoken,
        )
        self.engine = Pipeline.create_engine(
            onnx_file_path=onnx_file_path,
            engine_type=engine_type,
            engine_args=engine_args,
            context=engine_context,
        )
        self.multitoken = multitoken
        self.sampling_temperature = sampling_temperature
        self.deterministic = deterministic
        self.kv_cache_enabled = kv_cache_enabled
        self.kv_cache = DecoderKVCache(engine_type) if kv_cache_enabled else None

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
        self, inp: List[numpy.ndarray], val_inp: bool = True
    ) -> Tuple[int, numpy.ndarray]:
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

        if self.kv_cache:
            # if kv cache is enabled, we need to extract the kv cache state
            # from the output
            logits = self.extract_logits_from_output(out)
        else:
            logits = out[0]

        token = self.generate_token(logits=logits)

        return token, logits

    @staticmethod
    def overwrite_onnx_model_inputs(
        onnx_file_path: str,
        batch_size: int = 1,
        sequence_length: int = 128,
        multitoken: bool = False,
    ) -> Tuple[str, bool]:
        """
        Enforces the appropriate input shapes for the onnx model, as well as
        checks whether kv cache is enabled or not.

        :param onnx_file_path: The path to the onnx model file that will be
            overwritten with the new input shapes
        :param batch_size: The batch size to use for the input
        :param sequence_length: The sequence length to use for the input
        :param multitoken: Whether to run the engine in multitoken mode or not
        :return: The path to the onnx model file that has been overwritten
            with the new input shapes, as well as whether kv cache is enabled
            or not
        """
        model = onnx.load(onnx_file_path, load_external_data=False)
        initializer_input_names = set([node.name for node in model.graph.initializer])
        external_inputs = [
            inp for inp in model.graph.input if inp.name not in initializer_input_names
        ]
        input_names = []

        for external_input in external_inputs:
            # overwrite the batch size for all the inputs
            external_input.type.tensor_type.shape.dim[0].dim_value = batch_size

            if external_input.name in ["input_ids", "positions"]:
                external_input.type.tensor_type.shape.dim[1].dim_value = (
                    sequence_length if multitoken else 1
                )
            elif external_input.name == "attention_mask":
                # regardless of multitoken or not scenario, always provide full attention mask
                external_input.type.tensor_type.shape.dim[1].dim_value = sequence_length

            elif external_input.name.startswith(_CACHE_INPUT_NAME):
                # empty cache for multi-token runs,
                # otherwise max cache len is max len - 1
                external_input.type.tensor_type.shape.dim[2].dim_value = (
                    0 if multitoken else sequence_length - 1
                )
            else:
                raise ValueError(
                    f"Unexpected external input name: {external_input.name}"
                )
            input_names.append(external_input.name)

        _LOGGER.info(
            f"Overwriting in-place the input shapes of the transformer model at {onnx_file_path}"
        )
        save_onnx(model, onnx_file_path)

        is_cache_enabled = any(
            _CACHE_INPUT_NAME in node.name for node in model.graph.initializer
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
        # extend the engine input with the kv cache state
        kv_cache_state = self.kv_cache.cached_inputs
        if kv_cache_state is None:
            # if kv cache state is None, we need to initialize the
            # kv cache state and the session
            kv_cache_state = self._initialize_kv_cache_state(
                length=self.sequence_length if self.multitoken else 1
            )
            self.kv_cache.setup_session(
                session_id=generate_session_id(), state=kv_cache_state
            )
        # TODO: here we need to turn the kv cache state into a list of numpy arrays
        return inp.extend(kv_cache_state)

    def _extract_logits_from_output(self, out: List[numpy.ndarray]) -> numpy.ndarray:
        # extract the logits from the engine output
        # and update the kv cache state
        logits, *kv_cache_state = out
        self.kv_cache.update_session(
            state=kv_cache_state, ignore_generated=self.multitoken
        )
        return logits
